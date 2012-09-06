package net.electroland.gotham.core;

import java.awt.BorderLayout;
import java.io.IOException;
import java.util.logging.Logger;

import javax.swing.JFrame;

import net.electroland.gotham.core.ui.ELUControls;
import net.electroland.gotham.core.ui.RenderPanel;
import net.electroland.utils.lighting.ELUManager;

public class GothamConductor extends JFrame {

    static Logger logger = Logger.getLogger("GothamConductor");
    private static final long serialVersionUID = 6608878881526717236L;
    private RenderPanel renderPanel;
    private ELUManager lightingManager;

    public static void main(String[] args) throws IOException {

        GothamConductor conductor = new GothamConductor();

        conductor.lightingManager = new ELUManager();
        conductor.lightingManager.load(args.length > 0 ? args[0] : "lights.properties");
        conductor.configureRenderPanel(conductor.lightingManager);
        conductor.setSize(1200, 700);
        conductor.setTitle("Gotham Electroland Lighting Controls");
        conductor.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        conductor.setVisible(true);
    }

    public void configureRenderPanel(ELUManager lightingManager){
        ELUControls eluControls = new ELUControls(lightingManager);
        this.renderPanel = new RenderPanel(lightingManager);
        this.renderPanel.setLayout(new BorderLayout());
        eluControls.add(this.renderPanel.getDisplayControls());
        this.renderPanel.add(eluControls, BorderLayout.SOUTH);
        this.setContentPane(renderPanel);
    }
}