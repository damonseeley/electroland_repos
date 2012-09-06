package net.electroland.gotham.core.ui;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JPanel;

import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.DetectionModel;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.detection.BlueDetectionModel;
import net.electroland.utils.lighting.detection.GreenDetectionModel;
import net.electroland.utils.lighting.detection.RedDetectionModel;

public class RenderPanel extends JPanel implements Runnable {

    private static final long serialVersionUID = -3867812575633627878L;
    private ELUManager lightingManager;
    private DisplayControlBar controls;

    public RenderPanel(ELUManager lightingManager){
        this.lightingManager = lightingManager;
        this.controls = new DisplayControlBar();
    }

    public DisplayControlBar getDisplayControls(){
        return controls;
    }
    
    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        this.setBackground(Color.BLACK);
        Graphics2D g2d = (Graphics2D)g;

        // figure out which display we are rendering here.

        if (controls.includeRendering()){
            
        }
        if (controls.includePresenceGrid()){
            
        }
        if (controls.includeDectectors()){

            for (CanvasDetector cd : lightingManager.getCanvas(getSelectedCanvas()).getDetectors()){

                if (filtered(cd.getDetectorModel())){
                    g2d.setColor(getDetectorColor(cd, true));
                    g2d.fill(cd.getBoundary());
                    g2d.setColor(getDetectorColor(cd, false));
                    g2d.draw(cd.getBoundary());
                }
            }
        }
    }

    public boolean filtered(DetectionModel dm){
        if (controls.getDisplay().endsWith("RED")){
            return dm instanceof RedDetectionModel;
        } else if (controls.getDisplay().endsWith("GREEN")) {
            return dm instanceof GreenDetectionModel;
        } else if (controls.getDisplay().endsWith("BLUE")) {
            return dm instanceof BlueDetectionModel;
        } else {
            return false;
        }
    }
    
    public String getSelectedCanvas(){
        return controls.getDisplay().startsWith("East") ? "GothamEast" : "GothamWest";
    }
    
    public Color getDetectorColor(CanvasDetector cd, boolean incorporateState){
        int level = 255;
        if (incorporateState){
            level = (int)cd.getLatestState();
        }
        if (cd.getDetectorModel() instanceof RedDetectionModel){
            return new Color(level, 0, 0);
        } else if (cd.getDetectorModel() instanceof GreenDetectionModel) {
            return new Color(0, level, 0);
        } else if (cd.getDetectorModel() instanceof BlueDetectionModel) {
            return new Color(0, 0, level);
        } else {
            return new Color(level, level, level);
        }
    }
    
    @Override
    public void run() {
        while (true){
            repaint();
            try {
                Thread.sleep(33);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}