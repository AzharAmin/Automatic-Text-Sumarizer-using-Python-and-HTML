from flask import Flask, render_template,request,redirect,url_for,send_file
import Source.backup2 as summ
app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index1.html')

@app.route('/return-txt')
def return_files_tut():
		return send_file('result.txt', attachment_filename='summary.txt',as_attachment=True)

	
@app.route("/", methods=['POST'])
def getvalue():
	
	sentenceCount =int(request.form['lineCount'])
	textAbout = request.form['textAbout']
	textInput = request.form['textInput']

	with open("text.txt",'w' ,encoding='utf-8') as f:
		f.write(textInput)
		
	FS = summ.FuzzySummarizer()
	output=FS.Fuzzy1('text.txt',sentenceCount,textAbout)
	splitlist=output.split(".")
	
	with open("result.txt",'w',encoding='utf-8') as rs:
		for sent in splitlist:
			rs.write(sent+".")
			rs.write("\n")
	
	return render_template("output.html",summary=output)
	
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
	#serve(app, host='0.0.0.0', port=80)

 
