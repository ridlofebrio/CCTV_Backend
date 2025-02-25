from flask import Flask, send_file, render_template_string
import os
import mimetypes

app = Flask(__name__)


@app.route("/playback/<path:filename>")
def serve_video(filename):
    """Serve video files from Playback directory"""
    try:
        video_path = os.path.join("Playback", filename)

        if not os.path.exists(video_path):
            return f"Video not found: {filename}", 404

        # Set proper MIME type
        mimetype = mimetypes.guess_type(filename)[0] or "video/mp4"

        return send_file(
            video_path,
            mimetype=mimetype,
            as_attachment=False,  # Stream instead of download
            conditional=True,  # Support range requests
        )

    except Exception as e:
        return str(e), 500


@app.route("/")
def video_list():
    """Show list of available videos with players"""
    try:
        videos = [f for f in os.listdir("Playback") if f.endswith(".mp4")]

        return render_template_string(
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Video Playback</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        margin: 20px;
                        background: #f0f0f0;
                    }
                    .video-container {
                        margin-bottom: 30px;
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    video {
                        max-width: 100%;
                        height: auto;
                    }
                    h2 {
                        color: #333;
                        margin-bottom: 10px;
                    }
                </style>
            </head>
            <body>
                <h1>Available Videos</h1>
                {% for video in videos %}
                <div class="video-container">
                    <h2>{{ video }}</h2>
                    <video controls width="800">
                        <source src="{{ url_for('serve_video', filename=video) }}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
                {% endfor %}
            </body>
            </html>
        """,
            videos=videos,
        )

    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
