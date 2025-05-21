from flask import jsonify, Flask, request
from dotenv import load_dotenv
import os
from flask_cors import CORS
import requests
from requests.auth import HTTPBasicAuth



app = Flask(__name__)
CORS(app)
load_dotenv()


@app.route("/emp", methods=["POST"])
def emp():
    
    res = requests.get(
        "https://oic-axhgtg9tk9eg-je.integration.me-jeddah-1.ocp.oraclecloud.com/ic/api/integration/v1/flows/rest/EMP_D_INFO/1.0/employee",
        auth=HTTPBasicAuth(username, password),
        headers={"Accept": "application/json"}
    )
    employees = res.json()["items"]
    filters = request.get_json() # this is the user search
    actual = []


    for emp in employees:
        is_found = True
        for key, value in filters.items():
            if emp[key] != value:
                is_found = False
                break
        if is_found == True:
            actual.append(emp)
    return jsonify(actual)


# @app.route("/printEmp", methods=["POST"])
# def printEmp():
#     res = request.get_json()
#     emp()

# NO POINT OF THIS SINCE JUST CALL emp OF EMPLOYEE ID 
    
     # this will be used when u click on an employee from the table you should
         # be able to print the selected record

if __name__ == '__main__':
    app.run(debug=True)