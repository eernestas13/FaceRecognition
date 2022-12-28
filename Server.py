from Project import app
import face_recognition
from waitress import serve
import socket
import socketio

sio = socketio.Server()
server = socketio.WSGIApp(sio, app)
hostName = socket.gethostname()
ipAddress = socket.gethostbyname(hostName)

if __name__ == '__main__':
    serve(server, host='0.0.0.0', port=8080, url_scheme='http', threads=6)
