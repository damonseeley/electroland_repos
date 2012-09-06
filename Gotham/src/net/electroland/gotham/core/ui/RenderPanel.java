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
    }

    public void enableDisplayControls(){
        this.controls = new DisplayControlBar();
        this.add(controls);
    }
    
    @Override
    public void paint(Graphics g) {
        // TODO Auto-generated method stub
        super.paint(g);
    }

    @Override
    public void update(Graphics g) {
        // TODO Auto-generated method stub
        super.update(g);
    }
}