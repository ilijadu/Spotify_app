from flask import Flask, render_template, request
from funkcije import get_playlist_name,get_playlist_creator,button_scan,organize_button
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key="hello"

@app.route('/')
def index():
    return render_template("scan.html")

@app.route('/scan')
def scan():
    uri = request.args.get('uri')
    songs,df_without_lyrics,number_of_clusters,eval_metrics = button_scan(uri)

    plt.plot(eval_metrics.k, eval_metrics['tot.within.ss'], 'o-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')

    img_bytes.seek(0)
    plot_url = base64.b64encode(img_bytes.read())

    return render_template('results.html', songs=songs,name=get_playlist_name(uri),creator=get_playlist_creator(uri),
                           noLyrics=df_without_lyrics,nClusters=number_of_clusters,img_data=plot_url.decode('utf-8'))

@app.route('/organize')
def organize():
    nClust= int(request.args.get('clusters'))
    nTopics= int(request.args.get('topics'))

    number=organize_button(nClust,nTopics)

    return render_template('end.html',number=number)

@app.route('/end')
def end():
    return render_template('end.html')

if __name__ == '__main__':
    app.run(debug=True)