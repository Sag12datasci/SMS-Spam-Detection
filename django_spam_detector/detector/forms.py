from django import forms

class MessageForm(forms.Form):
    message = forms.CharField(
        widget=forms.Textarea(
            attrs={
                'class': 'form-control',
                'rows': 5,
                'placeholder': 'Enter your message here...'
            }
        ),
        required=True
    )
