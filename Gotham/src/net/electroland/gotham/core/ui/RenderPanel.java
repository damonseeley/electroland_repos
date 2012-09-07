package net.electroland.gotham.core.ui;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Ellipse2D;

import javax.swing.JPanel;

import net.electroland.utils.lighting.CanvasDetector;
import net.electroland.utils.lighting.DetectionModel;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.detection.BlueDetectionModel;
import net.electroland.utils.lighting.detection.GreenDetectionModel;
import net.electroland.utils.lighting.detection.RedDetectionModel;

public class RenderPanel extends JPanel implements Runnable, MouseListener, MouseMotionListener {

    private static final long serialVersionUID = -3867812575633627878L;
    private ELUManager lightingManager;
    private DisplayControlBar controls;
    final static int WAND_DIM = 100;

    public RenderPanel(ELUManager lightingManager){
        this.lightingManager = lightingManager;
        this.controls = new DisplayControlBar();
        new Thread(this).start();
        this.addMouseListener(this);
        this.addMouseMotionListener(this);
    }

    public DisplayControlBar getDisplayControls(){
        return controls;
    }
    
    @Override
    public void paintComponent(Graphics g) {

        super.paintComponent(g);
        this.setBackground(Color.BLACK);
        Graphics2D g2d = (Graphics2D)g;

        // wand for turning lights on and off.
        // TODO: this needs to be painted on the ELU Canvas, which in turn
        // needs to be painted here.  That will enable syncing.
        g2d.setColor(Color.WHITE);
        g2d.fill(new Ellipse2D.Double(mouseX, mouseY, WAND_DIM, WAND_DIM));

        // figure out which display we are rendering here.

        if (controls.includeRendering()){
            
        }
        if (controls.includePresenceGrid()){
            
        }
        if (controls.includeDectectors()){

            for (CanvasDetector cd : lightingManager.getCanvas(getSelectedCanvas()).getDetectors()){

                if (filtered(cd.getDetectorModel())){
                    g2d.setColor(getDetectorColor(cd));
                    g2d.fill(cd.getBoundary());
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
    
    public Color getDetectorColor(CanvasDetector cd){
        int level = (int)cd.getLatestState();

        if (level == 0){
            level = 150;
        } else {
            System.out.println("level" + level);
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
                Thread.sleep(33); // TODO: get from props
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    int mouseX, mouseY = -WAND_DIM;

    @Override
    public void mouseClicked(MouseEvent arg0) {
    }

    @Override
    public void mouseEntered(MouseEvent arg0) {
    }

    @Override
    public void mouseExited(MouseEvent arg0) {
        mouseX = mouseY = -WAND_DIM;
    }

    @Override
    public void mousePressed(MouseEvent arg0) {
    }

    @Override
    public void mouseReleased(MouseEvent arg0) {
        mouseX = mouseY = -WAND_DIM;
    }

    @Override
    public void mouseDragged(MouseEvent arg0) {
        mouseX = arg0.getX() - (WAND_DIM / 2);
        mouseY = arg0.getY() - (WAND_DIM / 2);
    }

    @Override
    public void mouseMoved(MouseEvent arg0) {
    }
}