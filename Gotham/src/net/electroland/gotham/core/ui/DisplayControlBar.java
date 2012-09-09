package net.electroland.gotham.core.ui;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class DisplayControlBar extends JPanel {

    private static final long serialVersionUID = -626583748057983309L;
    private JCheckBox includeRendering, includeDectectors, includePresenceGrid;
    private JComboBox detectorColors;

    public DisplayControlBar(){
        includeRendering    = new JCheckBox("canvas", true);
        includeDectectors   = new JCheckBox("detectors", true);
        includePresenceGrid = new JCheckBox("presence grid", true);
        detectorColors            = new JComboBox(new String[]{"red", "green", "blue", "composite"});
        this.add(new JLabel(" Display:"));
        this.add(detectorColors);
        this.add(includeRendering);
        this.add(includeDectectors);
        this.add(includePresenceGrid);
    }

    public boolean includeRendering(){
        return includeRendering.isSelected();
    }
    public boolean includeDectectors(){
        return includeDectectors.isSelected();
    }
    public boolean includePresenceGrid(){
        return includePresenceGrid.isSelected();
    }
    public String getDisplay(){
        return detectorColors.getSelectedItem().toString();
    }
}