package net.electroland.gotham.core.ui;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JPanel;

public class DisplayControlBar extends JPanel {

    private static final long serialVersionUID = -626583748057983309L;
    private JCheckBox includeRendering;
    private JCheckBox includeDectectors;
    private JCheckBox includePresenceGrid;
    private JComboBox displays;

    public DisplayControlBar(){
        displays            = new JComboBox(new String[]{"North face", "South face"});
        includeRendering    = new JCheckBox("Include rendering?", true);
        includeDectectors   = new JCheckBox("Include detectors?", true);
        includePresenceGrid = new JCheckBox("Include presence grid?", true);
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