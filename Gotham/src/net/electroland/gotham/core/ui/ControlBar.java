package net.electroland.gotham.core.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JPanel;
import javax.swing.event.ChangeEvent;

import net.miginfocom.swing.MigLayout;

public class ControlBar extends JPanel implements ActionListener, ButtonStateListener {

    private static final long serialVersionUID = -1589987975805310686L;
//    private JCheckBox includeRendering;
//    private JCheckBox includeDectectors;
//    private JCheckBox includePresenceGrid;
    private StatefulLabelButton startStop;
    private JButton allOn, allOff, sweep, trace;
    private JComboBox displays;
    private List<ControlBarListener> listeners;

    public ControlBar(){
        this.configureListeners();
        this.configureControls();
        this.layoutControls();
    }

    public void addListener(ControlBarListener listener){
        this.listeners.add(listener);
    }

    public void removeListener(ControlBarListener listener){
        this.listeners.remove(listener);
    }

    public void layoutControls(){
        this.setLayout(new MigLayout());
        this.add(startStop);
        this.add(allOn);
        this.add(allOff);
        this.add(sweep);
        this.add(trace);
        this.add(displays);
//        this.add(includeRendering);
//        this.add(includeDectectors);
//        this.add(includePresenceGrid);
    }

    public void configureControls(){

        startStop           = new StatefulLabelButton("Start", "Stop");
        allOn               = new JButton("All off");
        allOff              = new JButton("All on");
        
        // TODO: these should be a combo box with all tests and a "do it!" button
        sweep               = new JButton("Sweep");
        trace               = new JButton("Trace");
        displays            = new JComboBox(new String[]{"North face", "South face"});

        // TODO: move these closer to RenderPanel
//        includeRendering    = new JCheckBox("Include rendering?", true);
//        includeDectectors   = new JCheckBox("Include detectors?", true);
//        includePresenceGrid = new JCheckBox("Include presence grid?", true);

        allOn.addActionListener(this);
        allOff.addActionListener(this);
        sweep.addActionListener(this);
        trace.addActionListener(this);
        displays.addActionListener(this);
        startStop.addListener(this);
    }

    public void configureListeners(){
        listeners = new Vector<ControlBarListener>();
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == allOn){
            for (ControlBarListener l : listeners){
                l.allOn();
            }
        } else if (e.getSource() == allOff) {
            for (ControlBarListener l : listeners){
                l.allOff();
            }
        } else if (e.getSource() == sweep) {
            for (ControlBarListener l : listeners){
                l.run("sweep");
            }
        } else if (e.getSource() == trace) {
            for (ControlBarListener l : listeners){
                l.run("trace");
            }
        } else if (e.getSource() == displays) {
            for (ControlBarListener l : listeners){
                l.changeDisplay(displays.getSelectedItem().toString());
            }
        }
    }

    @Override
    public void buttonStateChanged(boolean isOn) {
        if (isOn)
        {
            for (ControlBarListener l : listeners){
                l.start();
            }
        }else{
            for (ControlBarListener l : listeners){
                l.stop();
            }
        }
    }
}