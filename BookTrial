import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by vagrant on 3/30/17.
  */
object BookTrial {

  def main(args: Array[String]) {
    def toInt(in: String): Int = {

      def retoption(): Option[Int] = {
        try {
          Some((if (in.length > 9) in.drop(in.length - 9) else if (in.length == 0) "0" else in).toInt)
        } catch {
          case e: NumberFormatException => None
        }
      }

      retoption() match {
        case Some(i) => i
        case None => 99999999
      }
    }


    val user=3270
    val conf = new SparkConf().setAppName("Recommendation App").setMaster("local")
    val sc = new SparkContext(conf)
    val directory = "/home/vagrant/data/book"
    val numPartitions = 20
    val ratingRDD = sc.textFile(directory + "/book-rating.csv")
    val bookRDD = sc.textFile(directory + "/book.csv")


    val rddOfRating = ratingRDD.map { line =>
      val fields = line.split(";")

      //(fields(0).toLong, Rating(fields(0).toInt, (if(fields(1).length > 9) fields(1).drop(fields(1).length - 9) else if (fields(1).length ==0) "0" else fields(1)).toInt, fields(2).toDouble))
      (fields(0).toLong, Rating(fields(0).toInt, toInt(fields(1)), fields(2).toDouble))
    }
    val top50BookIDs = rddOfRating.map { rating => rating._2.product }
      .countByValue()
      .toList
      .sortBy(-_._2)
      .take(50)
      .map { ratingData => ratingData._1 }

    val bookMap=       bookRDD.map { line =>
      val fields = line.split(";")

      (toInt(fields(0)), fields(1))
    }.collect().toMap

    val top10Books = top50BookIDs.filter(id => bookMap.contains(id))
      .map { bookId => (bookId, bookMap.getOrElse(bookId, "No Movie Found")) }
      .sorted
      .take(10)

    /*
    //    val listOFRating = top10Books.map { getRating => {
    //    val r = new scala.util.Random

          //println(s"Please Enter The Rating For Book ${getRating._2} From 1 to 5 [0 if not Seen]")
    //      Rating(0, getRating._1, r.nextInt(9).toLong)
    //    }
    //    }
    */
    var userrating = rddOfRating.filter(data => data._1 == 9999999)

    var listOFRating = userrating.map(data => data._2)


    if (listOFRating.count <= 0){
      listOFRating = sc.parallelize(top10Books.map { getRating => {
        val r = new scala.util.Random

        //println(s"Please Enter The Rating For Book ${getRating._2} From 1 to 5 [0 if not Seen]")
        Rating(0, getRating._1, r.nextInt(9).toLong)
      }
      })
    }

    //val topTenBooks = sc.parallelize(listOFRating)
    val topTenBooks = listOFRating

    val traingRating = rddOfRating.filter(data => data._1 < 4)
      .values
      .union(topTenBooks)
      .repartition(numPartitions)
      .persist()

    val testRating = rddOfRating.filter(data => data._1 > 6)
      .values
      .union(topTenBooks)
      .repartition(numPartitions)
      .persist()

    val validationRating = rddOfRating.filter(data => data._1 >= 4 && data._1 <= 6)
      .values
      .union(topTenBooks)
      .repartition(numPartitions)
      .persist()

    val numTraining = traingRating.count()
    val numTest = testRating.count()
    val numValidate = validationRating.count()
    print("My values to print " + numTraining + " " + numTest + " " + numValidate)

    val ratings = ratingRDD.map(_.split(";") match { case Array(user, item, rate) =>
      Rating(user.toInt, toInt(item), rate.toDouble)
    })

    val books = bookRDD.map { line =>
      val fields = line.split(";")

      (toInt(fields(0)), fields(1))
    }

    /*val books = bookRDD.map( _.split(";"))
      .map { case Array(bookid,bookname,author,year,publisher,images,imagem,imagel) => (toInt(bookid),bookname)
      }
*/

    val myRatingsRDD = topTenBooks
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

    val recommendedBooksId = model.predict(booksIHaveNotRead.map { product =>
      (0, product)}).map { case Rating(user,book,rating) => (book,rating) }
      .sortBy( x => x._2, ascending = false).take(20).map( x => x._1)

    println("Final Answer" + recommendedBooksId.length)

    recommendedBooksId.foreach(println)
    import java.io._
    val file = "/home/vagrant/data/output/" + user.toString + "_recommendation.txt"
    val writer = new BufferedWriter( new FileWriter(file))
    recommendedBooksId.foreach(x => writer.write(x.toString + "\n") )

    writer.close()
    //val file2 = "/home/vagrant/data/output/usercreated.txt"
    //new BufferedWriter( new FileWriter(file2))
    //val writer2 = new BufferedWriter( new FileWriter(file2))
    //listOFRating.foreach(x => writer.write(x.user + ":" + x.product + ":" + x.rating) )
    //writer2.close()
    sc.stop()

  }
}
