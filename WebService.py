from flask import Flask
from flask import jsonify
from RecommendationGeneratorNewCourses import genCourseRecommendation
from RecommendationGeneratorNewCourses import genLearningPathRecommendation

app = Flask(__name__)

@app.route("/recommendation/<userId>",methods=['GET'])

def welcome(userId):
    statement = """
    coursesrec: Generate Course and User description Recommenation \n 

    learningpathrec: Generate Learning Path related Reommendation"""
    return statement


@app.route("/recommendation/<userId>/coursesrec",methods=['GET'])
def getRecommendationDetails(userId):
    courseId = genCourseRecommendation(userId)
    return jsonify(courseId)

@app.route("/recommendation/<userId>/learningpathrec",methods=['GET'])
def getLearningRecommedationDetails(userId):
    courseId = genLearningPathRecommendation(userId)
    return jsonify(courseId)
    

if (__name__=="__main__"):
    app.run(port=5005)