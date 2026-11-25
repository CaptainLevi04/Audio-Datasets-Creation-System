# Kaggle Notebook Setup & Quickstart Guide

YouTube blocks direct downloads from Kaggle server IPs, so complete this **one-time setup** before running the notebook:

---

### 1. Export `cookies.txt` from Your Browser
1. Open **Chrome** or **Edge** in an **Incognito** window (prevents the session from expiring when closing your main browser). *(Optional -> you can open it in normal mode and not incognito)*
2. Go to [YouTube](https://www.youtube.com) and **Sign in** *(a secondary Google account is recommended for safety)*.
3. Open any video and play it for a second (forces YouTube to set active session cookies).
4. Install the **[Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)** extension *(enable **Allow in Incognito** in your browser's extension settings)*.
5. While on the YouTube tab, click the extension icon -> select **Current Site** -> click **Export** to download `cookies.txt`.
6. Close the Incognito window using the `X` button **without signing out**.

---

### 2. Upload Cookies to Kaggle as a Private Dataset
1. Open the notebook on Kaggle, go to the right sidebar, and click **Upload** -> **New Dataset**.

---

### 3. Configure Notebook Settings & Run
1. In the right sidebar (**Session options**):
   - Toggle **Internet** **ON** (required for `yt-dlp` to download the audio).
   - Select **Accelerator: GPU T4 x2** to run the transcription model.
2. Run