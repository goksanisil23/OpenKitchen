#!/usr/bin/env python3
import subprocess, shlex, sys, time, threading, signal
from Xlib import display, X
from Xlib.error import BadWindow
from PIL import Image


def find_window_by_pid(disp, pid, timeout=10.0, poll=0.2):
    root = disp.screen().root
    atom = disp.intern_atom("_NET_WM_PID")
    end = time.time() + timeout

    def walk(win):
        try:
            prop = win.get_full_property(atom, X.AnyPropertyType)
        except BadWindow:
            return None
        if prop and len(prop.value):
            try:
                if int(prop.value[0]) == pid:
                    return win
            except:
                pass
        try:
            for c in win.query_tree().children:
                r = walk(c)
                if r:
                    return r
        except BadWindow:
            pass
        return None

    while time.time() < end:
        w = walk(root)
        if w:
            return w
        time.sleep(poll)
    return None


def capture(win, out_pipe, fps):
    interval = 1.0 / fps
    while True:
        try:
            geom = win.get_geometry()
            w, h = geom.width, geom.height
            if w == 0 or h == 0:
                time.sleep(interval)
                continue
            raw = win.get_image(0, 0, w, h, X.ZPixmap, 0xFFFFFFFF)
            img = Image.frombytes("RGBA", (w, h), raw.data, "raw", "BGRA")
            rgb = img.convert("RGB")
            out_pipe.write(rgb.tobytes())
            time.sleep(interval)
        except BrokenPipeError:
            break
        except Exception:
            time.sleep(interval)


def main():
    if len(sys.argv) < 3:
        print("Usage: python auto_capture_simple.py '<command>' output.mp4 [fps]")
        sys.exit(1)
    cmd = sys.argv[1]
    out = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    proc = subprocess.Popen(shlex.split(cmd))
    print(f"Started app pid={proc.pid}")

    disp = display.Display()
    win = find_window_by_pid(disp, proc.pid, timeout=15.0)
    if not win:
        print("Window not found for PID; exiting.", file=sys.stderr)
        proc.terminate()
        sys.exit(1)

    geom = win.get_geometry()
    w, h = geom.width, geom.height
    if w == 0 or h == 0:
        print("Window has zero size.", file=sys.stderr)
        proc.terminate()
        sys.exit(1)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "slower",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        out,
    ]
    print("Launching ffmpeg:", " ".join(ffmpeg_cmd))
    ff = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    def watch():
        proc.wait()
        try:
            ff.stdin.close()
        except:
            pass

    threading.Thread(target=watch, daemon=True).start()

    def handle_sigint(signum, frame):
        proc.terminate()
        try:
            ff.stdin.close()
        except:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    capture(win, ff.stdin, fps)
    ff.wait()
    print("Done.")


if __name__ == "__main__":
    main()
