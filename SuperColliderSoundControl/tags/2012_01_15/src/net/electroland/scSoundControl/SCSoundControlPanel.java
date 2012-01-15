package net.electroland.scSoundControl;

import java.io.InputStream;

import javax.swing.BoxLayout;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

@SuppressWarnings("serial")
public class SCSoundControlPanel extends JPanel {

	private JTabbedPane _tabPane;
	public StreamedTextArea _scsynthOutputText;
	public SCSCStatsDisplay _statsDisplay;
	
	public SCSoundControlPanel() {
		//create a box layout.
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));

        _tabPane = new JTabbedPane();
        this.add(_tabPane);
        
        _scsynthOutputText = new StreamedTextArea(null);
        _statsDisplay = new SCSCStatsDisplay();
        _statsDisplay.init();
        
        _tabPane.add("scsynth", _scsynthOutputText);
        _tabPane.add("stats", _statsDisplay);
        
	}

	public void connectScsynthOutput(InputStream is) {
		_scsynthOutputText.setInputStream(is);
	}

}
