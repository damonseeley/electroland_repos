package net.electroland.broadcast.fish;

import java.util.ArrayList;

import junit.framework.TestCase;

public class PoolXMLSocketMessageTest extends TestCase {

	public void testToXML() {

		Fish a = new Fish("DWSDFBJKDLKDIWHDJLDK", 0, 1.0, 1.0, 1.0, 15.0, 10, 5.0, 0, 0, 45, null);
		Fish b = new Fish("SDJKFBKWJBHEOIUHDJKF", 0, 1.0, 100.0, 1.0, 39, 10, 5.0, 0, 0, 1, "myMove");
		Fish c = new Fish("SDFKJBWKJBEKJDKJBDFI", 0, 100.0, 1.0, 1.0, 45, 10, 5.0, 0, 0, 24, null);
		Fish d = new Fish("LKJDFPOIWEKNBRNMZKZD", 0, 100.0, 100.0, 1.0, -90, 10, 5.0, 0, 0, 99, null);

		ArrayList<Fish> list = new ArrayList<Fish>();
		list.add(a);
		list.add(b);
		list.add(c);
		list.add(d);

		PoolXMLSocketMessage mssg = new PoolXMLSocketMessage(list);
		System.out.println(mssg.toXML());
		/*
		<pool>
			<fish x="1" y="1" v="5" d="10" o="15" i="1" t="0" s="0" f="45"/>
			<fish x="1" y="100" v="5" d="10" o="39" i="2" t="0" s="0" f="1"/>
			<fish x="100" y="1" v="5" d="10" o="45" i="3" t="0" s="0" f="24"/>
			<fish x="100" y="100" v="5" d="10" o="-90" i="4" t="0" s="0" f="99"/>
		</pool>
		*/
	}
}
