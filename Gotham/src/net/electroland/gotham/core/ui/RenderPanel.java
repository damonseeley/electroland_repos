package net.electroland.gotham.core.ui;

import java.awt.Graphics;

import javax.swing.JPanel;

import net.electroland.utils.lighting.ELUManager;

public class RenderPanel extends JPanel {

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
    public void paint(Graphics g) {
        System.out.println("show:                " + controls.getDisplay());
        System.out.println("includeRendering:    " + controls.includeRendering());
        System.out.println("includeDectectors:   " + controls.includeDectectors());
        System.out.println("includePresenceGrid: " + controls.includePresenceGrid());
        super.paint(g);
    }

    @Override
    public void update(Graphics g) {
        // TODO Auto-generated method stub
        super.update(g);
    }
}