package net.electroland.gotham.core.ui;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class DisplayControlBar extends JPanel {

    private static final long serialVersionUID = -626583748057983309L;
    private JCheckBox includeRendering, includeDectectors, includePresenceGrid;
    private JComboBox displays;

    public DisplayControlBar(){
        includeRendering    = new JCheckBox("canas", true);
        includeDectectors   = new JCheckBox("detectors", true);
        includePresenceGrid = new JCheckBox("presence grid", true);
        displays            = new JComboBox(new String[]{"West RED", "West GREEN", "West BLUE", 
                                                         "East RED", "East GREEN", "East BLUE"});
        this.add(new JLabel(" Display:"));
        this.add(displays);
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
        return displays.getSelectedItem().toString();
    }
}