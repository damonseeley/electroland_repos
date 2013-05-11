package net.electroland.norfolk.core;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;

import javax.swing.JFrame;

import net.electroland.ea.Animation;
import net.electroland.eio.EIOManager;
import net.electroland.eio.IOFrameTest;
import net.electroland.eio.InputChannel;
import net.electroland.norfolk.core.viz.Raster2dViz;
import net.electroland.norfolk.core.viz.VizOSCSender;
import net.electroland.norfolk.eio.filters.PeopleIOWatcher;
import net.electroland.norfolk.eio.filters.PeopleListener;
import net.electroland.norfolk.eio.filters.PersonEvent;
import net.electroland.norfolk.sound.SimpleSoundManager;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ShutdownThread;
import net.electroland.utils.Shutdownable;
import net.electroland.utils.Util;
import net.electroland.utils.hours.OperatingHours;
import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.ELUCanvas;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.Fixture;
import net.electroland.utils.lighting.ui.ELUControls;

import org.apache.log4j.Logger;

public class Conductor implements PeopleListener, Runnable, Shutdownable{

    private static Logger       logger = Logger.getLogger(Conductor.class);
    private Animation           eam;
    private ELUManager          elu;
    private EIOManager          eio;
    private OperatingHours      hours;
    private ClipPlayer          clipPlayer;
    private VizOSCSender        viz;
    private Thread              thread;
    private int                 fps = 30;
    private JFrame              mainControls;
    private Raster2dViz         renderArea;
    private boolean             isHeadless = false;
    private static boolean      showSensors = false;
    private FpsAverage          fpsAvg = new FpsAverage(20);
    private Collection<Cue>     cues;
    private EventMetaData       meta;

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
            if (showSensors){
                IOFrameTest sensors = new IOFrameTest(c.eio);
                sensors.resizeWindow(1000, 200);
            }
        }

    }

    public void init() throws OptionException, IOException{

        elu = new ELUManager();
        elu.load(new ElectrolandProperties("norfolk-ELU2.properties"));

        eio = new EIOManager();
        eio.load(new ElectrolandProperties("io.properties"));

        PeopleIOWatcher pw = new PeopleIOWatcher();
        eio.addListener(pw);
        pw.addListener(this);

        ElectrolandProperties mainProps = new ElectrolandProperties("norfolk.properties");
        if (mainProps.getRequiredBoolean("settings", "global", "headless")) {
            isHeadless = true;
        }
        
        if (mainProps.getRequiredBoolean("settings", "global", "showsensors")) {
            showSensors = true;
        }

        eam = new Animation();
        eam.load(mainProps);
        eam.setBackground(Color.BLACK);
        fps = mainProps.getDefaultInt("settings", "global", "fps", 30);

        hours = new OperatingHours();
        hours.load(new ElectrolandProperties("hours.properties"));

        clipPlayer = new ClipPlayer(eam, new SimpleSoundManager(hours), elu, mainProps);
        new ClipPlayerGUI(clipPlayer);

        cues = new CueManager().load(mainProps);
        meta = new EventMetaData(30000); // TODO: load from props

        viz = new VizOSCSender();
        viz.load(mainProps);

        start();
    }

    public void start(){

        eio.start();

        if (thread == null){
            thread = new Thread(this);
            thread.start();
        }

        Runtime.getRuntime().addShutdownHook(new ShutdownThread(this));
    }

    public void stop(){
        elu.allOff();
        elu.stop();
        eio.shutdown();
        thread = null;
    }

    @Override
    public void run() {
        while (thread != null){


            // ELU and visualization rendering
            long startRender = System.currentTimeMillis();

            // Practically speaking, there's only one canvas, so we don't need
            // to do this iterator. Could just get it by name.
            for (ELUCanvas canvas : elu.getCanvases().values()){

                Dimension d = eam.getFrameDimensions();
                BufferedImage frame = eam.getFrame();

                // sync with ELU
                int pixels[] = new int[d.width * d.height];
                frame.getRGB(0, 0, d.width, d.height, pixels, 0, d.width);

                if (hours.shouldBeOpenNow("lights")){
                    CanvasDetector[] detectors = canvas.sync(pixels);

                    if (renderArea != null){
                        renderArea.update(frame, detectors, elu, fpsAvg.getAverage());
                        renderArea.repaint();
                    }

                    // sync with viz
                    if (viz.isEnabled()){
                        syncViz(detectors);
                    }

                }else{
                    elu.allOff();
                }

            }

            // Cues
            for (Cue c : cues){
                if (c.ready(meta) && !(c instanceof ChannelDriven)){
                    meta.addEvent(new CueEvent(c));
                    c.fire(meta, clipPlayer);
                }
            }

            // FPS management
            try{

                fpsAvg.touch();

                long currentTime = System.currentTimeMillis();
                int renderTime = (int)(currentTime - startRender);

                int sleepTime = (int)(1000.0/fps) - renderTime;
                if (sleepTime < 1){
                    sleepTime = 1;
                }

                Thread.sleep(sleepTime);

            }catch(InterruptedException e){
                logger.error(e);
            }
        }
    }


    // YUCK! (ELU actually has this already, and should allow getting it).
    private HashMap<String, Fixture> fixtures;
    private HashMap<String, Fixture> fixtureMap(){
        if (fixtures == null){
            fixtures = new HashMap<String, Fixture>();
            for (Fixture f : elu.getFixtures()){
                fixtures.put(f.getName(), f);
            }
        }
        return fixtures;
    }

    // very inefficient. would be nice if ELU merged RGB values smartly when
    // it has them all together.
    public void syncViz(CanvasDetector[] detectors){

        HashMap<String, RGB> fixtureColors = new HashMap<String, RGB>();

        for (CanvasDetector cd : detectors){

            String fixtureId = null;
            Integer r = null,g = null,b = null;
            for (String tag : cd.getTags()){
                if (tag.equals("red")){
                    r = Util.unsignedByteToInt(cd.getLatestState());
                } else if (tag.equals("green")){
                    g = Util.unsignedByteToInt(cd.getLatestState());
                } else if (tag.equals("blue")){
                    b = Util.unsignedByteToInt(cd.getLatestState());
                } else if (fixtureMap().containsKey(tag)){ // if this tag is a Fixture name
                    fixtureId = tag;
                }
            }

            // at this point we should have a fixture name, and a single color. need to put
            // it into fixtureColors, awaiting the other two colors.
            if (fixtureId != null){
                RGB rgb = fixtureColors.get(fixtureId);
                if (rgb == null){
                    rgb = new RGB();
                    fixtureColors.put(fixtureId, rgb);
                }
                if (r != null){
                    rgb.r = r;
                }
                if (g != null){
                    rgb.g = g;
                }
                if (b != null){
                    rgb.b = b;
                }
            }
        }

        // at this point fixtureColors should contain a color per fixture.
        HashMap<String, Color> dataToSend = new HashMap<String, Color>();

        for (String fixtureId : fixtureColors.keySet()){
            RGB rgb = fixtureColors.get(fixtureId);
            if (rgb != null){
                dataToSend.put(fixtureId, new Color(rgb.r, rgb.g, rgb.b));
            }
        }

        viz.setLights(dataToSend);
    }

    // used by syncViz only.
    class RGB{
        int r,g,b;
    }

    @Override
    public void personEntered(PersonEvent evt) {

        InputChannel channel = getChannel(evt.getChannelId());

        if (channel != null){
            for (Cue c : cues){
                // singlests and triplets
                if (c instanceof ChannelDriven && c.ready(meta)){
                    meta.addEvent(new SensorEvent()); // problem: ready test needs this.
                    ((ChannelDriven) c).fire(meta, clipPlayer, channel);
                }
            }
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

    @Override
    public void shutdown() {
        stop();
    }
}