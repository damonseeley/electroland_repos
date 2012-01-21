package net.electroland.edmonton.core;

import java.util.List;

import net.electroland.scSoundControl.SoundNode;

public class AudioFadeThread extends Thread {

    private List<SoundNode>nodes;
    private SoundController sc;
    private double duration;
    private long start;
    private int channels[];

    public AudioFadeThread(List<SoundNode>nodes, int channels[], SoundController sc, int duration){
        this.nodes = nodes;
        this.channels = channels;
        this.sc = sc;
        this.duration = duration;
        this.start = System.currentTimeMillis();
    }
    public void run() {
        boolean alive = true;
        while (alive)
        {
            float percentComplete = (System.currentTimeMillis() - start) / (float)duration;
            float level = 1.0f - 1.0f * percentComplete;
            if (percentComplete < 1.0){
                System.out.println(percentComplete);
                for (SoundNode node : nodes){
                    for (int i= 0; i < channels.length; i++)
                        node.setAmplitude(channels[i], level);
                }
            }else{
                for (SoundNode node : nodes){
                    sc.kill(node);
                }
                alive = false;
            }
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }
}