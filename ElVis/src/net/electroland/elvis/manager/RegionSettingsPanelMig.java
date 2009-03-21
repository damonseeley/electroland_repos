package net.electroland.elvis.manager;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.JSlider;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.miginfocom.swing.MigLayout;

public class RegionSettingsPanelMig implements Colorable {
	JLabel id;
	JSpinner depth;
	SpinnerNumberModel depthSpinnerModel;
	JTextField name;
	JSlider triggerPercentSlider;
	JLabel triggerPercentLabel;
	ColorButton colorButton;

	DecimalFormat format  ;


	public static RegionSettingsPanelMig thePanel;

	public RegionSettingsPanelMig() {
		thePanel = this;

		NumberFormat f = NumberFormat.getInstance();
		if (f instanceof DecimalFormat) {
			format = ((DecimalFormat) f);
			format.setPositivePrefix("");
			format.setMinimumIntegerDigits(3);
		}
	}

	public JPanel build() {
		JPanel p = new JPanel(new MigLayout("", "[left]"));

		p.add(new JLabel("ID:"),  "align right");
		id = new JLabel("id0");
		p.add(id,     "wrap");

		p.add(new JLabel("Name:"),  "align right");
		name = new JTextField("region00000000");
		name.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if(ImagePanel.THE_IMAGEPANEL.selectedRegion != null) {
					ImagePanel.THE_IMAGEPANEL.selectedRegion.name = name.getText();
				}				
			}
		});
		p.add(name, "");
		p.add(new JLabel("         "), "wrap");


		p.add(new JLabel("Trigger:"),  "align right");
		triggerPercentSlider = new JSlider(0,100,50);
		triggerPercentSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				triggerPercentLabel.setText(triggerPercentSlider.getValue() + "%");
				if(ImagePanel.THE_IMAGEPANEL.selectedRegion != null) {
					ImagePanel.THE_IMAGEPANEL.selectedRegion.percentage = ((float)triggerPercentSlider.getValue()) * .01f;
				}
			}

		});
		p.add(triggerPercentSlider, "");
		triggerPercentLabel = new JLabel("50%");
		p.add(triggerPercentLabel, "align left, wrap");



		p.add(new JLabel("Display"),"split, span, gaptop 100");
		p.add(new JSeparator(),       "growx, wrap, gaptop 100");

		p.add(new JLabel("Depth:"),   "align right");
		depthSpinnerModel = new SpinnerNumberModel(0, 0, 10, 1);
		depth = new JSpinner(depthSpinnerModel);
		depth.addChangeListener(
				new ChangeListener() {
					public void stateChanged(ChangeEvent e) {
						ImagePanel.THE_IMAGEPANEL.move(ImagePanel.THE_IMAGEPANEL.selectedRegion, ((Integer)depth.getValue()).intValue());

					}

				}
		);
		p.add(depth, "wrap");

		colorButton = new ColorButton(65,40,this);
		p.add(new JLabel(""),  "align right");
		p.add(colorButton, "wrap");
		return p;
	}

	public Color getColor() {
		if(ImagePanel.THE_IMAGEPANEL.selectedRegion != null) {
			return ImagePanel.THE_IMAGEPANEL.selectedRegion.getColor();
		} else {
			return null;
		}
	}

	public void setColor(Color c) {
		if(ImagePanel.THE_IMAGEPANEL.selectedRegion != null) {
			ImagePanel.THE_IMAGEPANEL.selectedRegion.setColor(c);
//			colorButton.setSize(200, 300);
			colorButton.repaint();

		}		
		ImagePanel.THE_IMAGEPANEL.repaint();


	}

	public void updateForNewDisplay() {
		if(ImagePanel.THE_IMAGEPANEL.selectedRegion != null) {
			colorButton.setColor(getColor());
			depthSpinnerModel.setMaximum(ImagePanel.THE_IMAGEPANEL.regions.size() -1);
			depth.setValue(new Integer(ImagePanel.THE_IMAGEPANEL.regions.indexOf(ImagePanel.THE_IMAGEPANEL.selectedRegion)));
			id.setText(Integer.toString(ImagePanel.THE_IMAGEPANEL.selectedRegion.id));;
			name.setText(ImagePanel.THE_IMAGEPANEL.selectedRegion.name);
			triggerPercentSlider.setValue((int) (ImagePanel.THE_IMAGEPANEL.selectedRegion.percentage * 100));
		} else {
			id.setText("   ");
			name.setText("             ");
		}
	}
}

