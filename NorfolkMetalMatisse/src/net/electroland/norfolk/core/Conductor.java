package net.electroland.norfolk.core;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.swing.JFrame;

import net.electroland.ea.Animation;
import net.electroland.eio.EIOManager;
import net.electroland.eio.IOFrameTest;
import net.electroland.eio.InputChannel;
import net.electroland.norfolk.eio.filters.PeopleIOWatcher;
import net.electroland.norfolk.eio.filters.PeopleListener;
import net.electroland.norfolk.eio.filters.PersonEvent;
import net.electroland.norfolk.sound.SimpleSoundManager;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.ELUCanvas;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.ui.ELUControls;

import org.apache.log4j.Logger;

public class Conductor implements PeopleListener, Runnable{

    private static Logger       logger = Logger.getLogger(Conductor.class);
    private Animation           eam;
    private ELUManager          elu;
    private EIOManager          eio;
    private ClipPlayer          clipPlayer;
    private Thread              thread;
    private int                 fps = 30;
    private JFrame              mainControls;
    private Raster2dViz         renderArea;
    private boolean             isHeadless = false;

    public static void main(String args[]) throws OptionException, IOException{

        Conductor c = new Conductor();
        c.init(); // need a way to turn multiple args into multiple props file names- or just put them all in one file?

        if (!c.isHeadless){
            c.mainControls = new JFrame();
            c.mainControls.setBackground(Color.DARK_GRAY);
            c.mainControls.setSize(c.eam.getFrameDimensions().width + 100, 
                                    c.eam.getFrameDimensions().width + 100);
            c.mainControls.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            // controls
            ELUControls eluControls = new ELUControls(c.elu);
            c.mainControls.setLayout(new BorderLayout());
            c.mainControls.add(eluControls, BorderLayout.PAGE_END);

            c.renderArea = new Raster2dViz();
            c.renderArea.setPreferredSize(c.eam.getFrameDimensions());
            c.mainControls.add(c.renderArea, BorderLayout.CENTER); 
            c.mainControls.setVisible(true);

            // sensors
            IOFrameTest sensors = new IOFrameTest(c.eio);
            sensors.resizeWindow(400, 400);

            // TODO: window for 3d rendering
        }

    }

    public void init() throws OptionException, IOException{

        elu = new ELUManager();
        elu.load(new ElectrolandProperties("norfolk-ELU2.properties"));

        eio = new EIOManager();
        eio.load(new ElectrolandProperties("io-local.properties"));

        PeopleIOWatcher pw = new PeopleIOWatcher();
        eio.addListener(pw);
        pw.addListener(this);

        ElectrolandProperties mainProps = new ElectrolandProperties("norfolk.properties");

        eam = new Animation();
        eam.load(mainProps);
        eam.setBackground(Color.BLACK);
        fps = mainProps.getDefaultInt("settings", "global", "fps", 30);

        clipPlayer = new ClipPlayer(eam, new SimpleSoundManager(), mainProps);

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

            // Practically speaking, there's only one canvas, so we don't need
            // to do this iterator. Could just get it by name.
            for (ELUCanvas canvas : elu.getCanvases().values()){

                Dimension d = eam.getFrameDimensions();
                BufferedImage frame = eam.getFrame();

                // sync with ELU
                int pixels[] = new int[d.width * d.height];
                frame.getRGB(0, 0, d.width, d.height, pixels, 0, d.width);
                CanvasDetector[] detectors = canvas.sync(pixels);

                if (renderArea != null){
                    renderArea.update(frame, detectors);
                    renderArea.repaint();
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

        InputChannel channel = getChannel(evt.getChannelId());
        if (channel != null){
            clipPlayer.play(channel);
        }
    }

    @Override
    public void personExited(PersonEvent evt) {
        // TODO whatever behavior matches a person exiting here.
    }

    private InputChannel getChannel(String id){
        for (InputChannel channel : eio.getInputChannels()){
            if (channel.getId().equals(id)){
                return channel;
            }
        }
        return null;
    }

}