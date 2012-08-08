package net.electroland.gateway.mediamap;


public class PlayThread extends Thread {

    private int idx;
    private MediaManager mmgr;

    public PlayThread(int idx, MediaManager mmgr){
        this.idx = idx;
        this.mmgr = mmgr;
    }

    @Override
    public void run() {
        super.run();

        float frames = MediaManager.FPS * MediaManager.CLIP_LENGTH_SECS;
        int played = 0;

        mmgr.setMedia(idx);

        while (played++ < frames){
            float percentComplete = played / frames;
            mmgr.setAlpha(MediaManager.EASING_F.valueAt(percentComplete, 0.0f, 1.0f));
            try {
                sleep(1000 / MediaManager.FPS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}