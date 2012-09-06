package net.electroland.gotham.core;

import java.awt.BorderLayout;
import java.io.IOException;
import java.util.logging.Logger;

import javax.swing.JFrame;

import net.electroland.gotham.core.ui.ControlBar;
import net.electroland.gotham.core.ui.ControlBarListener;
import net.electroland.gotham.core.ui.RenderPanel;
import net.electroland.utils.lighting.ELUManager;
import net.miginfocom.swing.MigLayout;

public class GothamConductor extends JFrame implements ControlBarListener {

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
        conductor.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        conductor.setVisible(true);
    }

    public void configureRenderPanel(ELUManager lightingManager){
        ControlBar controls = new ControlBar();
        controls.addListener(this);
        this.renderPanel = new RenderPanel(lightingManager);
        this.renderPanel.setLayout(new BorderLayout());
        this.renderPanel.add(controls, BorderLayout.SOUTH);
        this.setLayout(new MigLayout());
        this.setContentPane(renderPanel);
    }

    @Override
    public void allOn() {
        logger.info("all on");
        lightingManager.allOn();
    }

    @Override
    public void allOff() {
        logger.info("all off");
        lightingManager.allOff();
    }

    @Override
    public void start() {
        logger.info("start");
        lightingManager.start();
    }

    @Override
    public void stop() {
        logger.info("stop");
        lightingManager.stop();
    }

    @Override
    public void changeDisplay(String display) {
        logger.info("change display to " + display);
    }

    @Override
    public void run(String runner) {
        logger.info("run " + runner);
    }
}