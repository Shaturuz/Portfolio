package lntFinProj;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Random;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class Main extends JFrame implements ActionListener{
	
	
	
	JPanel topPanel, midPanel,
			titlePanel,
			insPanel, viewPanel, uptPanel, delPanel, exitPanel;
	JLabel titleLabel;
	JButton insMenu, viewMenu, uptMenu, delMenu, exit;
	
	Dimension dimensionSize = new Dimension(150, 20);
	
	void viewMenu()
	{
		setTitle("Menu  Utama");
		setVisible(true);
		setSize(500, 400);
		setResizable(false);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		
	}

	public Main() {
		
		// Top Panel
		topPanel = new JPanel(new GridLayout(2, 1));
		titlePanel = new JPanel();
		titleLabel = new JLabel("Menu Utama");
		titleLabel.setFont(new Font("Arial", Font.BOLD, 30));
		
		titlePanel.add(titleLabel);
		topPanel.add(titlePanel);
		
		// Mid Panel;
		midPanel = new JPanel(new GridLayout(6, 1));
		insPanel = new JPanel();
		viewPanel = new JPanel();
		uptPanel = new JPanel();
		delPanel = new JPanel();
		exitPanel = new JPanel();
		
		insMenu = new JButton("Insert Menu");
		viewMenu = new JButton("View Menu");
		uptMenu = new JButton("Update Menu");
		delMenu = new JButton("Delete Menu");
		exit = new JButton("Exit");
		
		insPanel.add(insMenu);
		viewPanel.add(viewMenu);
		uptPanel.add(uptMenu);
		delPanel.add(delMenu);
		exitPanel.add(exit);
		insMenu.addActionListener(this);
		viewMenu.addActionListener(this);
		uptMenu.addActionListener(this);
		delMenu.addActionListener(this);
		exit.addActionListener(this);
		
		midPanel.add(insPanel);
		midPanel.add(viewPanel);
		midPanel.add(uptPanel);
		midPanel.add(delPanel);
		midPanel.add(exitPanel);
		
		add(topPanel, BorderLayout.NORTH);
		add(midPanel, BorderLayout.CENTER);
		viewMenu();
		
	}

	public static void main(String[] args) {
		
		new Main();
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if(e.getSource() == insMenu) {
			
			System.out.println("insert");
			this.dispose();
			new Insert();
		}
		if(e.getSource() == viewMenu) {
			
			System.out.println("view");
			this.dispose();
			new View();
		}
		if(e.getSource() == uptMenu) {
	
			System.out.println("update");
			this.dispose();
			new Update();
		}
		if(e.getSource() == delMenu) {
	
			System.out.println("delete");
			this.dispose();
			new Delete();
		}
		
		if(e.getSource() == exit) {
			
			this.dispose();
		}
		
	}

}
