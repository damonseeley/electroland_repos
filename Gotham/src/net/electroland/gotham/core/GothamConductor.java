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

@SuppressWarnings("serial")
public class GothamConductor extends JFrame {

    public static void main(String[] args) throws IOException {

        GothamConductor conductor = new GothamConductor();

        ELUManager lightingManager = new ELUManager();
        lightingManager.load(args.length > 0 ? args[0] : "Gotham-ELU2.properties");

        conductor.configureRenderPanel(lightingManager);
        conductor.setSize(800, 100);
        conductor.setTitle("Gotham Electroland Lighting Controls");
        conductor.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        conductor.setVisible(true);

        conductor.configureUDPClients(lightingManager);
    }

    public void configureUDPClients(ELUManager lightingManager) throws SocketException{
        GothamPresenceGridUDPClient gridClient = new GothamPresenceGridUDPClient(3458);
        for (ELUCanvas c : lightingManager.getCanvases().values()){
            if (((ProcessingCanvas)c).getApplet() instanceof GothamPApplet){
                GothamPApplet g = (GothamPApplet)((ProcessingCanvas)c).getApplet();
                gridClient.addListener(g);
            }
        }

        gridClient.start();
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