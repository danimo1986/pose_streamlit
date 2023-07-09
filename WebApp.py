import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from PIL import Image

# 関数を作成してコードを整理する
def process_video(file):
    cap = cv2.VideoCapture(file.name)
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    landmark_x = [[] for _ in range(33)]
    landmark_y = [[] for _ in range(33)]
    timestamps = []

    pTime = 0
    stframe = st.empty()  # 空のコンテナを作成

    while True:
        success, img = cap.read()
        if not success:
            break

        if img is None:
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                landmark_x[id].append(cx)
                landmark_y[id].append(cy)
            timestamps.append(time.time())

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, f"FPS={int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # FPSを表示
        stframe.image(img)  # 空のコンテナにフレームを表示
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    return landmark_x, landmark_y

# メインのStreamlitアプリケーション
def main():
    st.title("Pose Estimation") 
    st.title("for Landmarks Trajectories") 
    # サイドバーに動画の数を選択するセレクトボックスを表示
    st.sidebar.title("Single validation or comparison")
    num_videos = st.sidebar.selectbox("Number of Videos", [1, 2])

    # サイドバーに画像を表示
    st.sidebar.title("Reference:")
    st.sidebar.title("Pose Landmarks Index")
    image_path = "pose_landmarks_index.png"
    image = Image.open(image_path)
    st.sidebar.image(image, use_column_width=True)

    if num_videos == 1:
        # 動画1のアップロード
        f1 = st.file_uploader('Upload File 1')

        if f1 is not None:
            # 動画1の座標の推移を取得
            st.subheader("Video 1")
            tmpfile1 = tempfile.NamedTemporaryFile(delete=False)
            tmpfile1.write(f1.getvalue())
            tmpfile1.close()
            landmark_x1, landmark_y1 = process_video(tmpfile1)

            # 選択されたランドマークをプロット
            selected_landmarks = st.multiselect('Select Landmarks to Plot', range(33))

            # グラフをプロット
            fig, ax = plt.subplots(figsize=(8, 8))
            for i in selected_landmarks:
                ax.plot(landmark_x1[i], landmark_y1[i], label=f'Landmark {i}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Landmark Trajectories (Video 1)')
            ax.legend()
            ax.invert_yaxis()
            st.pyplot(fig)

    elif num_videos == 2:
        # 動画1のアップロード
        f1 = st.file_uploader('Upload File 1')

        # 動画2のアップロード
        f2 = st.file_uploader('Upload File 2')

        if f1 is not None and f2 is not None:
            # 動画1の座標の推移を取得
            st.subheader("Video 1")
            tmpfile1 = tempfile.NamedTemporaryFile(delete=False)
            tmpfile1.write(f1.getvalue())
            tmpfile1.close()
            landmark_x1, landmark_y1 = process_video(tmpfile1)

            # 動画2の座標の推移を取得
            st.subheader("Video 2")
            tmpfile2 = tempfile.NamedTemporaryFile(delete=False)
            tmpfile2.write(f2.getvalue())
            tmpfile2.close()
            landmark_x2, landmark_y2 = process_video(tmpfile2)

            # 選択されたランドマークをプロット
            selected_landmarks = st.multiselect('Select Landmarks to Plot', range(33))

            # グラフをプロット
            fig, ax = plt.subplots(nrows=3, figsize=(8, 24))

            # 動画1のグラフをプロット
            ax[0].set_title('Landmark Trajectories (Video 1)')
            for i in selected_landmarks:
                ax[0].plot(landmark_x1[i], landmark_y1[i], label=f'Landmark {i}')
            ax[0].set_xlabel('X Coordinate')
            ax[0].set_ylabel('Y Coordinate')
            ax[0].legend()
            ax[0].invert_yaxis()

            # 動画2のグラフをプロット
            ax[1].set_title('Landmark Trajectories (Video 2)')
            for i in selected_landmarks:
                ax[1].plot(landmark_x2[i], landmark_y2[i], label=f'Landmark {i}')
            ax[1].set_xlabel('X Coordinate')
            ax[1].set_ylabel('Y Coordinate')
            ax[1].legend()
            ax[1].invert_yaxis()

            # 動画1と動画2のグラフを結合してプロット
            ax[2].set_title('Combined Landmark Trajectories')
            for i in selected_landmarks:
                ax[2].plot(landmark_x1[i], landmark_y1[i], label=f'Landmark {i} (Video 1)')
                ax[2].plot(landmark_x2[i], landmark_y2[i], label=f'Landmark {i} (Video 2)')
            ax[2].set_xlabel('X Coordinate')
            ax[2].set_ylabel('Y Coordinate')
            ax[2].legend()
            ax[2].invert_yaxis()

            # グラフを表示
            st.pyplot(fig)

if __name__ == '__main__':
    main()





