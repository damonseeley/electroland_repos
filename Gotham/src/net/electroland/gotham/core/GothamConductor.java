package net.electroland.gotham.core;

import java.awt.BorderLayout;
import java.io.IOException;
import java.net.SocketException;

import javax.swing.JFrame;

import net.electroland.gotham.core.ui.DisplayControlBar;
import net.electroland.gotham.processing.GothamPApplet;
import net.electroland.utils.lighting.ELUCanvas;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.canvas.ProcessingCanvas;
import net.electroland.utils.lighting.ui.ELUControls;

import org.apache.log4j.Logger;

public class GothamConductor extends JFrame {

    private static final long serialVersionUID = 6608878881526717236L;
    static Logger logger = Logger.getLogger(GothamConductor.class);
    private ELUManager lightingManager;

    public static void main(String[] args) throws IOException {

        GothamConductor conductor = new GothamConductor();

        conductor.lightingManager = new ELUManager();
        conductor.lightingManager.load(args.length > 0 ? args[0] : "lights.properties");

        conductor.configureRenderPanel(conductor.lightingManager);
        conductor.setSize(800, 100);
        conductor.setTitle("Gotham Electroland Lighting Controls");
        conductor.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        conductor.setVisible(true);

        conductor.configureUDPClient(conductor.lightingManager);
    }

    public void configureUDPClient(ELUManager lightingManager) throws SocketException{
        GothamPresenceGridUDPClient client = new GothamPresenceGridUDPClient(3458);
        for (ELUCanvas c : lightingManager.getCanvases().values()){
            GothamPApplet g = (GothamPApplet)((ProcessingCanvas)c).getApplet();
            client.addListener(g);
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
}