package net.electroland.norfolk.core;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.swing.JFrame;

import net.electroland.ea.Animation;
import net.electroland.ea.AnimationListener;
import net.electroland.ea.Clip;
import net.electroland.ea.Sequence;
import net.electroland.ea.easing.CubicOut;
import net.electroland.ea.easing.Linear;
import net.electroland.ea.easing.QuinticIn;
import net.electroland.eio.Coordinate;
import net.electroland.eio.EIOManager;
import net.electroland.eio.InputChannel;
import net.electroland.norfolk.eio.filters.PeopleIOWatcher;
import net.electroland.norfolk.eio.filters.PeopleListener;
import net.electroland.norfolk.eio.filters.PersonEvent;
import net.electroland.norfolk.sound.SimpleSoundManager;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.Util;
import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.ELUCanvas;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.ui.ELUControls;

import org.apache.log4j.Logger;

public class Conductor implements PeopleListener, Runnable, AnimationListener{

    private static Logger       logger = Logger.getLogger(Conductor.class);
    private int                 totalOccupants = 0;
    private Animation           eam;
    private ELUManager          elu;
    private EIOManager          eio;
    private SimpleSoundManager  ssm;
    private Thread              thread;
    private int                 fps = 30;
    private JFrame              mainControls;
    private boolean             isHeadless = false;

    public static void main(String args[]) throws OptionException, IOException{

        Conductor c = new Conductor();
        c.init(); // need a way to turn multiple args into multiple props file names- or just put them all in one file?

        if (!c.isHeadless){
            c.mainControls = new JFrame();
            c.mainControls.setSize(c.eam.getFrameDimensions());
            c.mainControls.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            // controls
            ELUControls eluControls = new ELUControls(c.elu);
            c.mainControls.setLayout(new BorderLayout());
            c.mainControls.add(eluControls, BorderLayout.SOUTH);
            c.mainControls.setVisible(true);

            // TODO: window for 3d rendering            
        }

    }

    public void init() throws OptionException, IOException{

        ssm = new SimpleSoundManager();
        ssm.load(new ElectrolandProperties("norfolk.properties"));

        elu = new ELUManager();
        elu.load(new ElectrolandProperties("norfolk-ELU2.properties"));
        // TODO: set local fps

        eio = new EIOManager();
        eio.load(new ElectrolandProperties("io-local.properties"));
        PeopleIOWatcher pw = new PeopleIOWatcher();
        eio.addListener(pw);
        pw.addListener(this);

        eam = new Animation();
        eam.load(new ElectrolandProperties("norfolk.properties"));
        eam.setBackground(Color.BLACK);

        start();
    }

    public void start(){

        eio.start();

        if (thread == null){
            thread = new Thread(this);
            thread.start();
        }
    }

    public void stop(){
        thread = null;
    }

    @Override
    public void run() {
        while (thread != null){
            for (ELUCanvas canvas : elu.getCanvases().values()){
                Dimension d = eam.getFrameDimensions();

                BufferedImage frame = eam.getFrame();

                // render on screen
                if (!isHeadless){

                    // render animation
                    if (mainControls != null && mainControls.getGraphics() != null && d != null){
                        mainControls.getGraphics().drawImage(frame, 0, 0, (int)d.getWidth(), (int)d.getHeight(), null);
                    }
                    // TODO: render FPS here
                    // TODO: set lets on 3d Viz
                }

                // render on lights
                int pixels[] = new int[d.width * d.height];
                frame.getRGB(0, 0, d.width, d.height, pixels, 0, d.width);
                CanvasDetector[] detectors = canvas.sync(pixels);
                
                if (!isHeadless){
                    for (CanvasDetector cd : detectors){
                        Rectangle r = (Rectangle)(cd.getBoundary());
                        int i = Util.unsignedByteToInt(cd.getLatestState());
                        Color c = new Color(i,i,i);
                        if (mainControls != null && mainControls.getGraphics() != null){
                            mainControls.getGraphics().setColor(c);
                            mainControls.getGraphics().fillRect(r.x, r.y, r.width, r.height);
                            mainControls.getGraphics().setColor(Color.WHITE);
                            mainControls.getGraphics().drawRect(r.x, r.y, r.width, r.height);
                        }
                    }
                }

            }
            try{
                Thread.sleep((long)(1000.0 / fps));
            }catch(InterruptedException e){
                logger.error(e);
            }
        }
    }

    @Override
    public void personEntered(PersonEvent evt) {

//        totalOccupants++; // we can't tell enter versus exit now.

        if (totalOccupants > 3){
            ssm.playSound("quack");
            // TODO: somekind of wave like effect

        }else{

            InputChannel channel = getChannel(evt.getChannelId());
            if (channel != null){

                // TODO: should switch case based on channel ID here (e.g., behavior per 
                // id that invokes a different method).

                ssm.playSound("001");

                Coordinate location = channel.getLocation();

                Clip one = eam.addClip(eam.getContent("slowImage"), (int)location.getX(), (int)location.getY(), 100, 100, 1.0f);

                Sequence bounce = new Sequence(); 

                bounce.yTo(150).yUsing(new QuinticIn()) // would be nice to make easing functions static.
                      .xBy(100).xUsing(new Linear())
                      .scaleWidth(2.0f)
                      .duration(1000)
               .newState()
                      .yTo(75).yUsing(new CubicOut())
                      .xBy(100).xUsing(new Linear())
                      .scaleWidth(.5f)
                      .duration(1000);

               // three bouncing clips:
               one.queue(bounce).queue(bounce).queue(bounce).fadeOut(500).deleteWhenDone();

                // TODO: render something location appropriate on the panel. 
            }
        }
    }

    private InputChannel getChannel(String id){
        for (InputChannel channel : eio.getInputChannels()){
            if (channel.getId().equals(id)){
                return channel;
            }
        }
        return null;
    }
    
    @Override
    public void personExited(PersonEvent evt) {
        totalOccupants--;
        // whatever behavior matches a person exiting here.
    }

    @Override
    public void messageReceived(Object message) {
        // TODO Auto-generated method stub
        // animation manager
    }
}