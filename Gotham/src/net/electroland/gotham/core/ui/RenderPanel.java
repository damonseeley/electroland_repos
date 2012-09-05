package net.electroland.gotham.core.ui;

import java.awt.BorderLayout;

import javax.swing.JPanel;

public class RenderPanel extends JPanel {

    private static final long serialVersionUID = -3867812575633627878L;

    public RenderPanel(){
        this.setLayout(new BorderLayout());
        this.add(new ControlBar(), BorderLayout.SOUTH);
    }
}