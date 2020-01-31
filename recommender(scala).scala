package book

import java.util.Scanner

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}



class BookRecommender extends java.io.Serializable {

  val conf = new SparkConf().setAppName("Recommendation App").setMaster("local")
  val sc = new SparkContext(conf)
  val directory = "/home/vagrant/data/book"
  val scanner = new Scanner(System.in)
  val numPartitions = 20
  val topTenBooks = getRatingFromUser
  val numTraining = getTrainingRating.count()
  val numTest = getTestingRating.count()
  val numValidate = getValidationRating.count()

  def getRatingRDD: RDD[String] = {

    sc.textFile(directory + "/book-rating.csv")
  }

  def getBookRDD: RDD[String] = {

    sc.textFile(directory + "/book.csv")
  }


  def toInt(in: String): Int= {

    def retoption(): Option[Int] = {
      try {
        Some((if(in.length > 9) in.drop(in.length - 9) else if (in.length ==0) "0" else in).toInt)
      } catch {
        case e: NumberFormatException => None
      }
    }

    retoption() match {
      case Some(i) => i
      case None => 99999999
    }
  }

  def getRDDOfRating: RDD[(Long, Rating)] = {

    getRatingRDD.map { line => val fields = line.split(";")

      //(fields(0).toLong, Rating(fields(0).toInt, (if(fields(1).length > 9) fields(1).drop(fields(1).length - 9) else if (fields(1).length ==0) "0" else fields(1)).toInt, fields(2).toDouble))
      (fields(0).toLong, Rating(fields(0).toInt, toInt(fields(1)), fields(2).toDouble))
    }
  }

  def getBookMap: Map[Int, String] = {

    getBookRDD.map { line => val fields = line.split(";")

      (fields(0).toInt, fields(1))
    }.collect().toMap
  }
//Without Recommendation it shows the top50 using RDD//
  def getTopTenBooks: List[(Int, String)] = {

    val top50BookIDs = getRDDOfRating.map { rating => rating._2.product }
      .countByValue()
      .toList
      .sortBy(-_._2)
      .take(50)
      .map { ratingData => ratingData._1 }

    top50BookIDs.filter(id => getBookMap.contains(id))
      .map { movieId => (movieId, getBookMap.getOrElse(movieId, "No Movie Found")) }
      .sorted
      .take(10)
  }

  def getRatingFromUser: RDD[Rating] = {

    val listOFRating = getTopTenBooks.map { getRating => {
      val r = new scala.util.Random

      //println(s"Please Enter The Rating For Book ${getRating._2} From 1 to 5 [0 if not Seen]")
      Rating(0, getRating._1, r.nextInt(9).toLong)
    }
    }
    sc.parallelize(listOFRating)
  }

  def getTrainingRating: RDD[Rating] = {

    getRDDOfRating.filter(data => data._1 < 4)
      .values
      .union(topTenBooks)
      .repartition(numPartitions)
      .persist()
  }

  def getValidationRating: RDD[Rating] = {

    getRDDOfRating.filter(data => data._1 >= 4 && data._1 <= 6)
      .values
      .union(topTenBooks)
      .repartition(numPartitions)
      .persist()
  }

  def getTestingRating: RDD[Rating] = {

    getRDDOfRating.filter(data => data._1 > 6)
      .values
      .union(topTenBooks)
      .repartition(numPartitions)
      .persist()
  }

/*  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {

    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }*/

}

object BookRecommender extends App {

  val bookRecommendationHelper = new BookRecommender with Serializable
  val sc = bookRecommendationHelper.sc
  // Load and parse the data
  val data = sc.textFile("/home/vagrant/data/book/book-rating.csv")
  val ratings = data.map(_.split(";") match { case Array(user, item, rate) =>
    Rating(user.toInt, (if(item.length > 9) item.drop(item.length - 9) else if (item.length ==0) "0" else item).toInt, rate.toDouble)
  })
  val books = bookRecommendationHelper.getBookRDD.map( _.split(";"))
    .map { case Array(bookid,bookname,genre) => (bookid.toInt ,bookname) }

  val myRatingsRDD = bookRecommendationHelper.topTenBooks
  val training = ratings.filter { case Rating(userId, bookId, rating) => (userId * bookId) % 10 <= 3 }.persist
  val test = ratings.filter { case Rating(userId, bookId, rating) => (userId * bookId) % 10 > 3}.persist

  val model = ALS.train(training.union(myRatingsRDD), 8, 10, 0.01)

  val booksIHaveRead = myRatingsRDD.map(x => x.product).collect().toList

  val booksIHaveNotRead = books.filter { case (movieId, name) => !booksIHaveRead.contains(movieId) }.map( _._1)

  val predictedRates =
    model.predict(test.map { case Rating(user,item,rating) => (user,item)} ).map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.persist()

  val ratesAndPreds = test.map { case Rating(user, product, rate) =>
    ((user, product), rate)
  }.join(predictedRates)

  val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) => Math.pow((r1 - r2), 2) }.mean()

  println("Mean Squared Error = " + MSE)

  val recommendedMoviesId = model.predict(booksIHaveNotRead.map { product =>
    (0, product)}).map { case Rating(user,movie,rating) => (movie,rating) }
    .sortBy( x => x._2, ascending = false).take(20).map( x => x._1)
  bookRecommendationHelper.sc.stop()
}