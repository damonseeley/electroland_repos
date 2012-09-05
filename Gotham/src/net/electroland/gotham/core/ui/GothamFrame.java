package net.electroland.gotham.core.ui;

import javax.swing.JFrame;

import net.miginfocom.swing.MigLayout;

public class GothamFrame extends JFrame {

    private static final long serialVersionUID = 5978286102899634536L;
    private RenderPanel renderPanel;

    public GothamFrame(){
        this.configureControls();
        this.layoutControls();
    }

    public void configureControls(){
        this.renderPanel = new RenderPanel();
    }
    public void layoutControls(){
        this.setLayout(new MigLayout());
        this.setContentPane(renderPanel);
    }
}