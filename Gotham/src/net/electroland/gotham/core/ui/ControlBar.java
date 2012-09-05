package net.electroland.gotham.core.ui;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JPanel;

import net.miginfocom.swing.MigLayout;

public class ControlBar extends JPanel {

    private static final long serialVersionUID = -1589987975805310686L;
    private JCheckBox includeRendering;
    private JCheckBox includeDectectors;
    private JCheckBox includePresenceGrid;
    private StatefulLabelButton startStop;
    private JButton allOn, allOff, sweep, trace;
    private JComboBox displays;

    public ControlBar(){
        this.configureControls();
        this.layoutControls();
    }

    public void layoutControls(){
        this.setLayout(new MigLayout());
        this.add(startStop);
        this.add(allOn);
        this.add(allOff);
        this.add(sweep);
        this.add(trace);
        this.add(displays);
        this.add(includeRendering);
        this.add(includeDectectors);
        this.add(includePresenceGrid);
    }

    public void configureControls(){
        allOn               = new JButton("All off");
        allOff              = new JButton("All on");
        sweep               = new JButton("Sweep");
        trace               = new JButton("Trace");
        includeRendering    = new JCheckBox("Include rendering?", true);
        includeDectectors   = new JCheckBox("Include detectors?", true);
        includePresenceGrid = new JCheckBox("Include presence grid?", true);
        startStop           = new StatefulLabelButton("Start", "Stop");
        displays            = new JComboBox(new String[]{"North face", "South face"});
    }
}