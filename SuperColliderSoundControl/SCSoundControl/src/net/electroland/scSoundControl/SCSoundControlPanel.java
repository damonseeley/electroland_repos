package net.electroland.scSoundControl;

import java.io.InputStream;

import javax.swing.BoxLayout;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

public class SCSoundControlPanel extends JPanel {

	JTabbedPane _tabPane;
	StreamedTextArea _scsynthOutputText;
	
	public SCSoundControlPanel() {
		//create a box layout.
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));

        _tabPane = new JTabbedPane();
        this.add(_tabPane);
        
        _scsynthOutputText = new StreamedTextArea(null);
        
        _tabPane.add("scsynth", _scsynthOutputText);
        _tabPane.add("stats", new JPanel());
        
	}

	public void connectScsynthOutput(InputStream is) {
		_scsynthOutputText.setInputStream(is);
	}

}
