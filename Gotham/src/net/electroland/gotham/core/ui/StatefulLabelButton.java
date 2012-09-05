package net.electroland.gotham.core.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;

public class StatefulLabelButton extends JButton implements ActionListener{

    private static final long serialVersionUID = 9109315371948848914L;
    private String onLabel, offLabel;
    boolean isOn;

    public StatefulLabelButton(String offLabel, String onLabel){

        this.isOn = false;
        this.onLabel = onLabel;
        this.offLabel = offLabel;
        this.addActionListener(this);

        setState(isOn);
        setText();
    }

    public boolean isOn(){
        return isOn;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        isOn = !isOn;
        setText();
    }

    public void setText(){
        super.setText(isOn ? onLabel : offLabel);
    }

    public void setState(boolean isOn){
        this.isOn = isOn;
        setText();
    }

    final public void setText(String text){
        throw new RuntimeException("Attempt to change label from outside StatefulLabelButton");
    }
}