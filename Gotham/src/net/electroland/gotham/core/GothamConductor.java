package net.electroland.gotham.core;

import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.SocketException;

import javax.swing.JFrame;

import net.electroland.gotham.core.ui.DisplayControlBar;
import net.electroland.gotham.processing.GothamPApplet;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.lighting.ELUCanvas;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.canvas.ProcessingCanvas;
import net.electroland.utils.lighting.ui.ELUControls;
import net.electroland.utils.process.ProcessExitedListener;
import net.electroland.utils.process.ProcessItem;
import net.electroland.utils.process.ProcessUtil;

import org.apache.log4j.Logger;

public class GothamConductor extends JFrame implements ProcessExitedListener {

    private static final long serialVersionUID = 6608878881526717236L;
    static Logger logger = Logger.getLogger(GothamConductor.class);
    private ELUManager lightingManager;
    
    public static ElectrolandProperties props;
    

    public static void main(String[] args) throws IOException {

        GothamConductor conductor = new GothamConductor();

        conductor.initProps();

        conductor.lightingManager = new ELUManager();
        conductor.lightingManager.load(args.length > 0 ? args[0] : "Gotham-ELU2.properties");

        conductor.configureRenderPanel(conductor.lightingManager);
        conductor.setSize(800, 100);
        conductor.setTitle("Gotham Electroland Lighting Controls");
        conductor.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        conductor.setVisible(true);

        conductor.configureUDPClients(conductor.lightingManager);

        conductor.startElVis();
    }

    public void configureUDPClients(ELUManager lightingManager) throws SocketException{
        GothamPresenceGridUDPClient gridClient = new GothamPresenceGridUDPClient(3458);
        for (ELUCanvas c : lightingManager.getCanvases().values()){
            if (((ProcessingCanvas)c).getApplet() instanceof GothamPApplet){
                GothamPApplet g = (GothamPApplet)((ProcessingCanvas)c).getApplet();
                gridClient.addListener(g);
            }
        }
/**        
        GothamRegionUDPClient regionClient = new GothamRegionUDPClient(3457);
        //add listeners!!!
        
        
        GothamTrackUDPClient trackClient = new GothamTrackUDPClient(3456);
        for (ELUCanvas c : lightingManager.getCanvases().values()){
            if (((ProcessingCanvas)c).getApplet() instanceof GothamPApplet){
                GothamPApplet g = (GothamPApplet)((ProcessingCanvas)c).getApplet();
                trackClient.addListener(g);
            }
        }
 */
        
        //Start the clients up
        //trackClient.start();
        gridClient.start();
        //regionClient.start();
    }
    
    public void initProps() {
        props = new ElectrolandProperties("Gotham-global.properties");
    }

    public void startElVis(){
        String batFileName = props.getOptional("elvis", "run", "bat_file");
        if (batFileName != null){
            String rootDir = props.getRequired("elvis", "run", "run_dir");
            int pollRate = props.getRequiredInt("elvis", "run", "poll_rate");
            ProcessUtil.run(readBat(batFileName, rootDir), new File(rootDir), this, pollRate);
        }
    }

    public String readBat(String filename, String rootDir){
        if (filename == null){
            return null;
        }else{
            File bat = new File(rootDir, filename);
            BufferedReader br;
            try {
                br = new BufferedReader(new FileReader(bat));
                String command = br.readLine();
                br.close();
                return command;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public void configureRenderPanel(ELUManager lightingManager){
        ELUControls eluControls = new ELUControls(lightingManager);
        DisplayControlBar displayControls = new DisplayControlBar();
        for (ELUCanvas c : lightingManager.getCanvases().values()){
            if (c instanceof ProcessingCanvas){
                displayControls.addListener(((ProcessingCanvas)c).getApplet());
            }
        }
        eluControls.add(displayControls);
        this.setLayout(new BorderLayout());
        this.add(eluControls, BorderLayout.SOUTH);
        this.add(displayControls, BorderLayout.NORTH);
    }

    @Override
    public void exited(ProcessItem arg0) {
        startElVis();
    }
}