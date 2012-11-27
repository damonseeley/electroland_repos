package net.electroland.norfolk.eio.analog;

import java.awt.Color;
import java.awt.Graphics;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import javax.swing.JFrame;

import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;

@SuppressWarnings("serial")
public class Test extends JFrame{

    private ModbusTCPMaster connection;

    public static void main(String[] args) throws Exception {

        Test t = new Test();
        t.setSize(600, 400);
        t.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        t.setVisible(true);
        while (true){
            t.repaint();
            Thread.sleep(33);
        }
    }

    @Override
    public void paint(Graphics g){

        InputRegister[] data;

        g.setColor(Color.BLACK);
        g.fillRect(0, 0, this.getWidth(), this.getHeight());
        g.setColor(Color.WHITE);

        try {

            if (connection == null){
                connection = new ModbusTCPMaster("192.168.247.21");
                connection.connect();
            }

            data = connection.readInputRegisters(192, 2);
            int y = 50;
            short lastShort = -1;
            System.out.println();
            for (InputRegister register : data){

                ByteBuffer be = ByteBuffer.allocate(2);
                be.order(ByteOrder.BIG_ENDIAN);
                for (byte b : register.toBytes()){
                    be.put(b);
                }
                short beVal = be.getShort(0);
                if (lastShort == -1){
                    lastShort = beVal;
                }else{
                    if (beVal > lastShort){
                        g.setColor(Color.RED);
                    }
                }
                int width = scale(beVal, this.getWidth());
                g.fillRect(0, y+=100, width, 50);
            }

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public int scale(short value, int dim){
        float percent = value / (float)Short.MAX_VALUE;
        return (int)(percent * dim);
    }
}