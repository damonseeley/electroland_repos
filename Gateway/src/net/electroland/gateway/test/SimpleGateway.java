package net.electroland.gateway.test;

import net.electroland.gateway.mediamap.MediaManager;
import oscP5.OscBundle;
import oscP5.OscEventListener;
import oscP5.OscMessage;
import oscP5.OscP5;
import oscP5.OscStatus;

public class SimpleGateway implements OscEventListener{

    public static void main(String args[]){

        OscP5 oscP5 = MediaManager.configureOSC(new SimpleGateway());
        int fps = Integer.parseInt(args[0]);

        System.out.println("length: " + args.length);
        int[] ids = new int[args.length - 1];
        for (int i = 1; i < args.length; i++){
            System.out.println(Integer.parseInt(args[i]));
            ids[i-1] = Integer.parseInt(args[i]);
            System.out.println(ids[i-1]);
        }

        int currentId = 0;

        while (true){

            // alpha fade ON layer 1
            boolean first = true;
            for (float alpha = 0.0f; alpha < 1.0f; alpha+= (1.0f/fps)/3){
                if (first){
                    OscMessage setMedia = new OscMessage(MediaManager.SET_MEDIA_2);
                    System.out.println("send SET_MEDIA " + ids[currentId]);
                    System.out.println("send SET_ALPHA " + alpha);
                    setMedia.add(ids[currentId++]);
                    OscMessage setAlpha = new OscMessage(MediaManager.SET_ALPHA_1);
                    setAlpha.add(alpha);

                    OscBundle bndl = new OscBundle();
                    bndl.add(setMedia);
                    bndl.add(setAlpha);
                    oscP5.send(bndl);
                    if (currentId == ids.length){
                        currentId = 0;
                    }
                    first = false;

                }else{
                    OscMessage setAlpha = new OscMessage(MediaManager.SET_ALPHA_1);
                    setAlpha.add(alpha);
                    oscP5.send(setAlpha);
                    System.out.println("send SET_ALPHA " + alpha);
                    sleep(1000/fps);
                }
            }

            // set layer 2
            first = true;
            for (float alpha = 1.0f; alpha > 0.0f; alpha-= (1.0f/fps)/3){
                if (first){
                    OscMessage setMedia = new OscMessage(MediaManager.SET_MEDIA_1);
                    System.out.println("send SET_MEDIA " + ids[currentId]);
                    System.out.println("send SET_ALPHA " + alpha);
                    setMedia.add(ids[currentId++]);
                    OscMessage setAlpha = new OscMessage(MediaManager.SET_ALPHA_1);
                    setAlpha.add(alpha);

                    OscBundle bndl = new OscBundle();
                    bndl.add(setMedia);
                    bndl.add(setAlpha);
                    oscP5.send(bndl);
                    if (currentId == ids.length){
                        currentId = 0;
                    }
                    first = false;

                }else{
                    OscMessage setAlpha = new OscMessage(MediaManager.SET_ALPHA_1);
                    setAlpha.add(alpha);
                    oscP5.send(setAlpha);
                    System.out.println("send SET_ALPHA " + alpha);
                    sleep(1000/fps);
                }
            }
        }
    }

    public static void sleep(long time){
        try {
            Thread.sleep(time);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void oscEvent(OscMessage arg0) {
        System.out.println("received event: " + arg0);
    }

    @Override
    public void oscStatus(OscStatus arg0) {
        System.out.println("received status: " + arg0);
    }
}