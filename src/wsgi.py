from data_app import app

server = app.server

if __name__ == "__main__":
    app.server.run(debug=True)
