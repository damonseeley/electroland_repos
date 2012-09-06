package net.electroland.gotham.core.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;

import net.electroland.utils.lighting.ELUManager;

public class ELUControls extends JPanel implements ActionListener, ButtonStateListener {

    private static final long serialVersionUID = -1589987975805310686L;
    private ELUManager lightingManager;
    private StatefulLabelButton startStop;
    private JButton allOn, allOff, runTest;
    private JComboBox tests;
    // TODO: add widgets for "reload" and fps display

    public ELUControls(ELUManager lightingManager){

        this.lightingManager = lightingManager;

        this.configureControls();
        this.layoutControls();
    }

    public void layoutControls(){
        this.add(new JLabel("ELU:"));
        this.add(startStop);
        this.add(allOn);
        this.add(allOff);
        this.add(new JLabel("|"));
        this.add(tests);
        this.add(runTest);
    }

    public void configureControls(){

        startStop = new StatefulLabelButton("Start", "Stop");
        allOn     = new JButton("All off");
        allOff    = new JButton("All on");
        runTest   = new JButton("Run test");
        tests     = new JComboBox(lightingManager.getTestSuites());

        allOn.addActionListener(this);
        allOff.addActionListener(this);
        runTest.addActionListener(this);
        startStop.addListener(this);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == allOn){
            lightingManager.allOn();
        } else if (e.getSource() == allOff) {
            lightingManager.allOff();
        } else if (e.getSource() == runTest) {
            lightingManager.runTest(tests.getSelectedItem().toString());
        }
    }

    @Override
    public void buttonStateChanged(StatefulLabelButton button) {
        if (button == startStop && button.isOn()) {
            lightingManager.start();
        } else {
            lightingManager.stop();
        }
    }
}