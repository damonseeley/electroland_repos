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
    private boolean isProcessing = false;
    // TODO: add widgets for "reload" and fps display

    public ELUControls(ELUManager lightingManager, boolean isProcessing){

        this.lightingManager = lightingManager;
        this.isProcessing = isProcessing;

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

        startStop = new StatefulLabelButton("start", "stop");
        allOn     = new JButton("all on");
        allOff    = new JButton("all off");
        runTest   = new JButton("run test");
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
            if (isProcessing){
                lightingManager.pstart();
            }else{
                lightingManager.start();
            }
        } else {
            if (isProcessing){
                lightingManager.pstop();
            }else{
                lightingManager.stop();
            }
        }
    }
}