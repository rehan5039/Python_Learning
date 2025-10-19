import json
import os
from datetime import datetime, timedelta

class Book:
    def __init__(self, isbn, title, author, genre, total_copies=1):
        self.isbn = isbn
        self.title = title
        self.author = author
        self.genre = genre
        self.total_copies = total_copies
        self.available_copies = total_copies
    
    def to_dict(self):
        return {
            'isbn': self.isbn,
            'title': self.title,
            'author': self.author,
            'genre': self.genre,
            'total_copies': self.total_copies,
            'available_copies': self.available_copies
        }
    
    @classmethod
    def from_dict(cls, data):
        book = cls(
            data['isbn'],
            data['title'],
            data['author'],
            data['genre'],
            data['total_copies']
        )
        book.available_copies = data['available_copies']
        return book

class Member:
    def __init__(self, member_id, name, email, phone):
        self.member_id = member_id
        self.name = name
        self.email = email
        self.phone = phone
        self.borrowed_books = []  # List of borrowed book ISBNs
        self.fines = 0.0
    
    def to_dict(self):
        return {
            'member_id': self.member_id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'borrowed_books': self.borrowed_books,
            'fines': self.fines
        }
    
    @classmethod
    def from_dict(cls, data):
        member = cls(
            data['member_id'],
            data['name'],
            data['email'],
            data['phone']
        )
        member.borrowed_books = data['borrowed_books']
        member.fines = data['fines']
        return member

class Transaction:
    def __init__(self, transaction_id, member_id, isbn, transaction_type, date=None):
        self.transaction_id = transaction_id
        self.member_id = member_id
        self.isbn = isbn
        self.transaction_type = transaction_type  # 'borrow' or 'return'
        self.date = date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.due_date = None
        if transaction_type == 'borrow':
            due_date_obj = datetime.now() + timedelta(days=14)
            self.due_date = due_date_obj.strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self):
        return {
            'transaction_id': self.transaction_id,
            'member_id': self.member_id,
            'isbn': self.isbn,
            'transaction_type': self.transaction_type,
            'date': self.date,
            'due_date': self.due_date
        }
    
    @classmethod
    def from_dict(cls, data):
        transaction = cls(
            data['transaction_id'],
            data['member_id'],
            data['isbn'],
            data['transaction_type'],
            data['date']
        )
        transaction.due_date = data['due_date']
        return transaction

class Library:
    def __init__(self, data_file='library_data.json'):
        self.data_file = data_file
        self.books = {}      # isbn -> Book
        self.members = {}    # member_id -> Member
        self.transactions = []  # List of Transaction objects
        self.load_data()
    
    def load_data(self):
        """Load library data from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                # Load books
                for isbn, book_data in data.get('books', {}).items():
                    self.books[isbn] = Book.from_dict(book_data)
                
                # Load members
                for member_id, member_data in data.get('members', {}).items():
                    self.members[member_id] = Member.from_dict(member_data)
                
                # Load transactions
                for trans_data in data.get('transactions', []):
                    self.transactions.append(Transaction.from_dict(trans_data))
                    
                print("Library data loaded successfully!")
            except Exception as e:
                print(f"Error loading data: {e}")
        else:
            print("No existing data file found. Starting with empty library.")
    
    def save_data(self):
        """Save library data to JSON file"""
        try:
            data = {
                'books': {isbn: book.to_dict() for isbn, book in self.books.items()},
                'members': {member_id: member.to_dict() for member_id, member in self.members.items()},
                'transactions': [trans.to_dict() for trans in self.transactions]
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            print("Library data saved successfully!")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def add_book(self, isbn, title, author, genre, copies=1):
        """Add a new book to the library"""
        if isbn in self.books:
            # If book exists, increase copies
            self.books[isbn].total_copies += copies
            self.books[isbn].available_copies += copies
            print(f"Added {copies} more copies of '{title}'. Total copies: {self.books[isbn].total_copies}")
        else:
            # Create new book
            book = Book(isbn, title, author, genre, copies)
            self.books[isbn] = book
            print(f"Added new book: '{title}' by {author}")
        
        self.save_data()
    
    def remove_book(self, isbn):
        """Remove a book from the library"""
        if isbn in self.books:
            book = self.books[isbn]
            # Check if book is currently borrowed
            if book.available_copies < book.total_copies:
                print(f"Cannot remove book. {book.total_copies - book.available_copies} copies are currently borrowed.")
                return False
            
            del self.books[isbn]
            print(f"Removed book: '{book.title}'")
            self.save_data()
            return True
        else:
            print("Book not found!")
            return False
    
    def register_member(self, member_id, name, email, phone):
        """Register a new library member"""
        if member_id in self.members:
            print("Member ID already exists!")
            return False
        
        member = Member(member_id, name, email, phone)
        self.members[member_id] = member
        print(f"Registered new member: {name}")
        self.save_data()
        return True
    
    def remove_member(self, member_id):
        """Remove a member from the library"""
        if member_id in self.members:
            member = self.members[member_id]
            # Check if member has borrowed books
            if member.borrowed_books:
                print(f"Cannot remove member. They have {len(member.borrowed_books)} borrowed books.")
                return False
            
            del self.members[member_id]
            print(f"Removed member: {member.name}")
            self.save_data()
            return True
        else:
            print("Member not found!")
            return False
    
    def borrow_book(self, member_id, isbn):
        """Allow a member to borrow a book"""
        # Check if member exists
        if member_id not in self.members:
            print("Member not found!")
            return False
        
        # Check if book exists
        if isbn not in self.books:
            print("Book not found!")
            return False
        
        member = self.members[member_id]
        book = self.books[isbn]
        
        # Check if book is available
        if book.available_copies <= 0:
            print("No copies available for borrowing!")
            return False
        
        # Check if member has reached borrowing limit (5 books)
        if len(member.borrowed_books) >= 5:
            print("Member has reached the borrowing limit (5 books)!")
            return False
        
        # Check if member already borrowed this book
        if isbn in member.borrowed_books:
            print("Member has already borrowed this book!")
            return False
        
        # Process borrowing
        book.available_copies -= 1
        member.borrowed_books.append(isbn)
        
        # Create transaction
        transaction_id = len(self.transactions) + 1
        transaction = Transaction(transaction_id, member_id, isbn, 'borrow')
        self.transactions.append(transaction)
        
        print(f"{member.name} borrowed '{book.title}' (Due: {transaction.due_date})")
        self.save_data()
        return True
    
    def return_book(self, member_id, isbn):
        """Allow a member to return a book"""
        # Check if member exists
        if member_id not in self.members:
            print("Member not found!")
            return False
        
        # Check if book exists
        if isbn not in self.books:
            print("Book not found!")
            return False
        
        member = self.members[member_id]
        book = self.books[isbn]
        
        # Check if member actually borrowed this book
        if isbn not in member.borrowed_books:
            print("Member hasn't borrowed this book!")
            return False
        
        # Process return
        book.available_copies += 1
        member.borrowed_books.remove(isbn)
        
        # Calculate fine if overdue
        fine = 0
        for transaction in reversed(self.transactions):
            if (transaction.member_id == member_id and 
                transaction.isbn == isbn and 
                transaction.transaction_type == 'borrow'):
                
                due_date = datetime.strptime(transaction.due_date, "%Y-%m-%d %H:%M:%S")
                return_date = datetime.now()
                
                if return_date > due_date:
                    days_overdue = (return_date - due_date).days
                    fine = days_overdue * 1.0  # $1 per day overdue
                    member.fines += fine
                    print(f"Book is {days_overdue} days overdue. Fine: ${fine:.2f}")
                break
        
        # Create transaction
        transaction_id = len(self.transactions) + 1
        transaction = Transaction(transaction_id, member_id, isbn, 'return')
        self.transactions.append(transaction)
        
        print(f"{member.name} returned '{book.title}'")
        if fine > 0:
            print(f"Total fines for {member.name}: ${member.fines:.2f}")
        
        self.save_data()
        return True
    
    def search_books(self, query):
        """Search books by title, author, or genre"""
        results = []
        query = query.lower()
        
        for book in self.books.values():
            if (query in book.title.lower() or 
                query in book.author.lower() or 
                query in book.genre.lower()):
                results.append(book)
        
        return results
    
    def display_books(self):
        """Display all books in the library"""
        if not self.books:
            print("No books in the library!")
            return
        
        print("\n=== Library Books ===")
        for book in self.books.values():
            status = f"{book.available_copies}/{book.total_copies} available"
            print(f"ISBN: {book.isbn}")
            print(f"  Title: {book.title}")
            print(f"  Author: {book.author}")
            print(f"  Genre: {book.genre}")
            print(f"  Copies: {status}")
            print()
    
    def display_members(self):
        """Display all library members"""
        if not self.members:
            print("No members registered!")
            return
        
        print("\n=== Library Members ===")
        for member in self.members.values():
            borrowed_count = len(member.borrowed_books)
            print(f"ID: {member.member_id}")
            print(f"  Name: {member.name}")
            print(f"  Email: {member.email}")
            print(f"  Phone: {member.phone}")
            print(f"  Books Borrowed: {borrowed_count}")
            print(f"  Outstanding Fines: ${member.fines:.2f}")
            print()
    
    def display_member_books(self, member_id):
        """Display books borrowed by a specific member"""
        if member_id not in self.members:
            print("Member not found!")
            return
        
        member = self.members[member_id]
        if not member.borrowed_books:
            print(f"{member.name} has not borrowed any books.")
            return
        
        print(f"\n=== Books Borrowed by {member.name} ===")
        for isbn in member.borrowed_books:
            if isbn in self.books:
                book = self.books[isbn]
                # Find due date from transactions
                due_date = "Unknown"
                for transaction in reversed(self.transactions):
                    if (transaction.member_id == member_id and 
                        transaction.isbn == isbn and 
                        transaction.transaction_type == 'borrow'):
                        due_date = transaction.due_date
                        break
                print(f"  - {book.title} by {book.author} (Due: {due_date})")
    
    def pay_fines(self, member_id, amount):
        """Allow a member to pay their fines"""
        if member_id not in self.members:
            print("Member not found!")
            return False
        
        member = self.members[member_id]
        if amount > member.fines:
            print(f"Amount exceeds outstanding fines. Enter ${member.fines:.2f} or less.")
            return False
        
        member.fines -= amount
        print(f"Payment of ${amount:.2f} received. Remaining fines: ${member.fines:.2f}")
        self.save_data()
        return True

def main():
    """Main function to run the library management system"""
    library = Library()
    
    while True:
        print("\n=== Library Management System ===")
        print("1. Add Book")
        print("2. Remove Book")
        print("3. Register Member")
        print("4. Remove Member")
        print("5. Borrow Book")
        print("6. Return Book")
        print("7. Search Books")
        print("8. Display All Books")
        print("9. Display All Members")
        print("10. Display Member's Books")
        print("11. Pay Fines")
        print("12. Exit")
        
        try:
            choice = int(input("\nEnter your choice (1-12): "))
            
            if choice == 1:
                # Add Book
                isbn = input("Enter ISBN: ")
                title = input("Enter title: ")
                author = input("Enter author: ")
                genre = input("Enter genre: ")
                copies = int(input("Enter number of copies (default 1): ") or "1")
                library.add_book(isbn, title, author, genre, copies)
            
            elif choice == 2:
                # Remove Book
                isbn = input("Enter ISBN of book to remove: ")
                library.remove_book(isbn)
            
            elif choice == 3:
                # Register Member
                member_id = input("Enter member ID: ")
                name = input("Enter member name: ")
                email = input("Enter email: ")
                phone = input("Enter phone number: ")
                library.register_member(member_id, name, email, phone)
            
            elif choice == 4:
                # Remove Member
                member_id = input("Enter member ID to remove: ")
                library.remove_member(member_id)
            
            elif choice == 5:
                # Borrow Book
                member_id = input("Enter member ID: ")
                isbn = input("Enter ISBN of book to borrow: ")
                library.borrow_book(member_id, isbn)
            
            elif choice == 6:
                # Return Book
                member_id = input("Enter member ID: ")
                isbn = input("Enter ISBN of book to return: ")
                library.return_book(member_id, isbn)
            
            elif choice == 7:
                # Search Books
                query = input("Enter search query (title/author/genre): ")
                results = library.search_books(query)
                if results:
                    print(f"\nFound {len(results)} book(s):")
                    for book in results:
                        status = f"{book.available_copies}/{book.total_copies} available"
                        print(f"  - {book.title} by {book.author} ({status})")
                else:
                    print("No books found matching your query.")
            
            elif choice == 8:
                # Display All Books
                library.display_books()
            
            elif choice == 9:
                # Display All Members
                library.display_members()
            
            elif choice == 10:
                # Display Member's Books
                member_id = input("Enter member ID: ")
                library.display_member_books(member_id)
            
            elif choice == 11:
                # Pay Fines
                member_id = input("Enter member ID: ")
                amount = float(input("Enter amount to pay: $"))
                library.pay_fines(member_id, amount)
            
            elif choice == 12:
                # Exit
                print("Thank you for using the Library Management System!")
                break
            
            else:
                print("Invalid choice! Please enter a number between 1-12.")
        
        except ValueError:
            print("Invalid input! Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()