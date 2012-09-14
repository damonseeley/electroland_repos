package net.electroland.gotham.core.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileNameExtensionFilter;

import net.electroland.gotham.processing.GothamPApplet;
import net.electroland.utils.lighting.DetectionModel;
import net.electroland.utils.lighting.canvas.ELUPApplet;
import net.electroland.utils.lighting.detection.BlueDetectionModel;
import net.electroland.utils.lighting.detection.GreenDetectionModel;
import net.electroland.utils.lighting.detection.RedDetectionModel;

public class DisplayControlBar extends JPanel implements ActionListener, ChangeListener{

    private static final long serialVersionUID = -626583748057983309L;
    private JCheckBox includeRendering, includeDectectors;//, includePresenceGrid;
    private JComboBox detectorColors, detectorScale;
    private JButton one, two;
    private Vector<ELUPApplet> listeners;

    public DisplayControlBar(){
        includeRendering    = new JCheckBox("canvas", true);
        includeDectectors   = new JCheckBox("detectors", true);
        detectorColors      = new JComboBox(new DetectionModel[]{null, new RedDetectionModel(), new BlueDetectionModel(), new GreenDetectionModel()});
        detectorScale       = new JComboBox(new Float[]{1.0f, 2.0f, 5.0f, 10.0f});
        one                 = new JButton("load one");
        two                 = new JButton("load two");
        this.add(new JLabel(" Display:"));
        this.add(detectorColors);
        this.add(includeDectectors);
        this.add(new JLabel("at scale:"));
        this.add(detectorScale);
        this.add(includeRendering);
        this.add(one);
        this.add(two);
        this.listeners = new Vector<ELUPApplet>();
        detectorColors.addActionListener(this);
        detectorScale.addActionListener(this);
        includeRendering.addChangeListener(this);
        includeDectectors.addChangeListener(this);
        one.addActionListener(this);
        two.addActionListener(this);
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
        } else if (evt.getSource() == one) {
            File f = getFilename();
            if (f != null) {
                ((GothamPApplet)listeners.get(0)).fileReceived(f);
            }
        } else if (evt.getSource() == two) {
            File f = getFilename();
            if (f != null) {
                ((GothamPApplet)listeners.get(1)).fileReceived(f);
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

    public File getFilename() {
        JFileChooser chooser = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
            "JPG, PNG or MOV Files", "jpg", "mov", "png");
        chooser.setFileFilter(filter);
        int returnVal = chooser.showOpenDialog(this);
        if(returnVal == JFileChooser.APPROVE_OPTION) {
            return chooser.getSelectedFile();
        }
        else return null;
    }
}