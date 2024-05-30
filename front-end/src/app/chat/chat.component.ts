import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatMessage } from '../ChatMessage';
import { NgClass } from '@angular/common';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    FormsModule,
    NgClass
  ],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css',
})

export class ChatComponent {
  message: string = '';
  isLoading: boolean = false;
  chatMessages: ChatMessage[] = [
    {
      message: 'Hello! How can I help you today?',
      messageCounterPosition: 0,
      isUser: false
    }
  ];

  async sendMessage() {
    // Strip the message of only the last \n
    this.message = this.message.replace(/\n$/, '')
    if (this.message === '') return;
    this.isLoading = true;

    //Flask API 127.0.0.7:5000
    const prompt : JSON = <JSON><unknown>{
      "prompt": this.message
    };

    this.addMessage(this.message, true);

    //Clear text area
    this.message = '';
    //Post request to the Flask API
    let response = await fetch('http://127.0.0.1:5000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': 'http://127.0.0.1:5000/chat'
      },
      body: JSON.stringify(prompt)
    });

    let data = await response.json();    
    let answer: string = data['response'];

    this.addMessage(answer, false);
    this.isLoading = false;

  }

  addMessage(message: string, isUser: boolean) {
    this.chatMessages.push({
      message: message,
      messageCounterPosition: this.chatMessages.length,
      isUser: isUser
    });
  }


  
}
