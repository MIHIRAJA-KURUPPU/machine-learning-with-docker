from flask import Flask, request

app = Flask(__name__)

# Add the missing decorator symbol '@'
@app.route('/')
def add():
    a = request.args.get("a")
    b = request.args.get("b")
    return str(int(a) + int(b))

if __name__ == '__main__':
    app.run(port=7000)


#localhost:7000/?a=10&b=20