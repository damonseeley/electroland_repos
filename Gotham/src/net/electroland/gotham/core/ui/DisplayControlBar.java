package net.electroland.gotham.core.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.electroland.utils.lighting.DetectionModel;
import net.electroland.utils.lighting.canvas.ELUPApplet;
import net.electroland.utils.lighting.detection.BlueDetectionModel;
import net.electroland.utils.lighting.detection.GreenDetectionModel;
import net.electroland.utils.lighting.detection.RedDetectionModel;

public class DisplayControlBar extends JPanel implements ActionListener, ChangeListener{

    private static final long serialVersionUID = -626583748057983309L;
    private JCheckBox includeRendering, includeDectectors;//, includePresenceGrid;
    private JComboBox detectorColors, detectorScale;
    private Vector<ELUPApplet> listeners;

    public DisplayControlBar(){
        includeRendering    = new JCheckBox("canvas", true);
        includeDectectors   = new JCheckBox("detectors", true);
        //includePresenceGrid = new JCheckBox("presence grid", true);
        detectorColors            = new JComboBox(new DetectionModel[]{null, new RedDetectionModel(), new BlueDetectionModel(), new GreenDetectionModel()});
        detectorScale            = new JComboBox(new Float[]{1.0f, 2.0f, 5.0f, 10.0f});
        this.add(new JLabel(" Display:"));
        this.add(detectorColors);
        this.add(includeDectectors);
        this.add(new JLabel("at scale:"));
        this.add(detectorScale);
        this.add(includeRendering);
        //this.add(includePresenceGrid);
        this.listeners = new Vector<ELUPApplet>();
        detectorColors.addActionListener(this);
        detectorScale.addActionListener(this);
        includeRendering.addChangeListener(this);
        includeDectectors.addChangeListener(this);
    }

    public void addListener(ELUPApplet applet){
        listeners.add(applet);
    }
    public void removeListener(ELUPApplet applet){
        listeners.remove(applet);
    }

    @Override
    public void actionPerformed(ActionEvent evt) {
        if (evt.getSource() == detectorColors){
            for (ELUPApplet a : listeners){
                if (detectorColors.getSelectedItem() instanceof DetectionModel){
                    a.showOnly((DetectionModel)detectorColors.getSelectedItem());
                }else{
                    a.showAll();
                }
            }
        } else if (evt.getSource() == detectorScale){
            for (ELUPApplet a : listeners){
                a.setDetectorScale((Float)detectorScale.getSelectedItem());
            }
        }
    }

    @Override
    public void stateChanged(ChangeEvent evt) {
        if (evt.getSource() == includeRendering) {
            for (ELUPApplet a : listeners){
                a.setShowRendering(includeRendering.isSelected());
            }
        } else if (evt.getSource() == includeDectectors) {
            for (ELUPApplet a : listeners){
                a.setShowDetectors(includeDectectors.isSelected());
                detectorColors.setEnabled(includeDectectors.isSelected());
            }
        }
    }
}