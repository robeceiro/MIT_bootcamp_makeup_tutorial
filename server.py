from flask import Flask
import flask
import os
app = flask.Flask(__name__)

@app.route("/")
def start():
   data = {"success":True}
   cmd = 'sudo sh -c "echo \'0\' > /sys/class/backlight/rpi_backlight/bl_power"'
   print(cmd)
   os.system(cmd)
   #subprocess.check_output([cmd])
   return flask.jsonify(data)

@app.route("/stop")
def stop():
   data = {"success":True}
   cmd = 'sudo sh -c "echo \'1\' > /sys/class/backlight/rpi_backlight/bl_power"'
   print(cmd)
   os.system(cmd)
   #subprocess.check_output([cmd])
   return flask.jsonify(data)

# start()

app.run()
