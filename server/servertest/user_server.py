from flask import Flask, request,send_file
import time

app = Flask(__name__)

@app.route("/get_json")
def get_json():
    return send_file("match.json")

@app.route("/submit", methods=["POST"])
def submit():
    if "result" in request.files:
        file = request.files["result"]
        file.save("result.json")
        print("result.json を保存しました")

    # "answer" というキーで送られてきたファイルを保存
    if "numnum" in request.files:
        file = request.files["numnum"]
        file.save("numnum.json")
        print("numnum.json を保存しました")

    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
