from flask import Flask, request, jsonify, Response, url_for
from flask_restful import Resource, Api
from pymongo import MongoClient
from werkzeug.security import generate_password_hash,check_password_hash
import json
import requests
import classify_image 

app = Flask(__name__)
api = Api(app)
client = MongoClient("mongodb://db:27017")
UserDB = client["UserDB"]
User = UserDB["User"]
AdminDB = client["AdminDB"] #new database for Admins
Admin = AdminDB["Admin"] #new collection called Admin

Admin.insert_one({"adminname" : "henry", "password" : "something"})
class Register(Resource):
    def post(self):
        user_data = request.get_json()
        username = user_data["username"]
        password = user_data["password"]

        if len(username) < 4:
            jsontext = {
                "message" : "Username must be at least 4 characters",
                "status" : 301
            }
            return jsonify(jsontext)
        elif len(password) < 4:
            jsontext = {
                "message" : "Password must be at least 4 characters",
                "status" : 301
            }
            return jsonify(jsontext)
        elif User.count_documents({"username" : username}) != 0:
            jsontext = {
                "message" : "This username is already created",
                "status" : 301
            }
            return jsonify(jsontext)
        else:
            #add users' usernames and passwords to the database
            User.insert_one({
                "username" : username,
                "password" : generate_password_hash(password,method = "sha256"),
                "tokens" : 10,
                "text1" : [],
                "text2" : [] 
            })

            jsontext = {
                "message" : "Account created successfully",
                "status" : 200
            }
            return jsonify(jsontext)

class Classify(Resource):
    def post(self):
        user_data = request.get_json()
        url = user_data["url"]

        jsontext = classify_image.classify(url)
        return jsonify(jsontext)
        
        



class Refill(Resource):
    def post(self):
        data = request.get_json()
        username = data["username"]
        admin_password = data["admin_password"]
        refill_amount = data["refill_amount"]

        if User.count_documents({"username" : username}) == 0:
            jsontext = {
                "message" : "Account does not exist",
                "status" : 301
            }
            return jsonify(jsontext)
        elif Admin.count_documents({"password" : admin_password}) == 0:
            jsontext = {
                "message" : "Refill not authorized",
                "status" : 304
            }
            return jsonify(jsontext)
        else:
            new_tokens = User.find({"username":username})[0]["tokens"] + refill_amount
            User.update_one(
                {"username" : username},
                {"$set" : {"tokens" : new_tokens}}
            )
            jsontext = {
                "message" : "Refilled successfully",
                "status" : 200,
                "current_tokens": new_tokens
            }
            return jsonify(jsontext)







api.add_resource(Register,"/register")
api.add_resource(Classify,"/classify")
api.add_resource(Refill,"/refill")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")