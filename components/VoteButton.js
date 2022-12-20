import { firestore, auth, increment } from '@lib/firebase';
import { useDocument } from 'react-firebase-hooks/firestore';

// Allows user to Vote or like a post
export default function Vote({ postRef }) {
  // Listen to Vote document for currently logged in user
  const VoteRef = postRef.collection('Votes').doc(auth.currentUser.uid);
  const [VoteDoc] = useDocument(VoteRef);

  // Create a user-to-post relationship
  const addVote = async () => {
    const uid = auth.currentUser.uid;
    const batch = firestore.batch();

    batch.update(postRef, { VoteCount: increment(1) });
    batch.set(VoteRef, { uid });

    await batch.commit();
  };

  const addDownVote = async () => {
    const uid = auth.currentUser.uid;
    const batch = firestore.batch();

    batch.update(postRef, { VoteCount: increment(-1) });
    batch.set(VoteRef, { uid });

    await batch.commit();
  };

  // Remove a user-to-post relationship
  const removeVote = async () => {
    const batch = firestore.batch();

    batch.update(postRef, { VoteCount: increment(-1) });
    batch.delete(VoteRef);

    await batch.commit();
  };

  return VoteDoc?.exists ? (
    <button className="block-button" onClick={removeVote}>Remove Vote</button>
  ) : [
    <button className="block-button" onClick={addVote}>Up</button>,
    <button className="block-button" onClick={addDownVote}>Down</button>
  ];
}
