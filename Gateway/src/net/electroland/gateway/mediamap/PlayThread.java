package net.electroland.gateway.mediamap;

import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;

public class PlayThread extends Thread {

    private String idx;

    public PlayThread(String idx){
        this.idx = idx;
    }

    @Override
    public void run() {
        super.run();
        // 33 fps * 2 seconds
        float frames = GenerateDB.fps * GenerateDB.clipLengthSecs;
        int played = 0;

        EasingFunction ef = new Linear();

        setMedia(idx);

        while (played++ < frames){
            float percentComplete = played / frames;
            setAlpha(ef.valueAt(percentComplete, 1.0f, 0.0f));
            try {
                sleep(1000 / GenerateDB.fps);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public void setMedia(String idx){
        System.out.println("playing " + idx);
    }

    public void setAlpha(float alpha){
        System.out.println("  alpha to " + alpha);
    }
}