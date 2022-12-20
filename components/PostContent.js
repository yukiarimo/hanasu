import Link from 'next/link';
import ReactMarkdown from 'react-markdown';
import { firestore, auth, serverTimestamp } from '@lib/firebase';
import { useRouter } from 'next/router';
import toast from 'react-hot-toast';


// UI component for main post content
export default function PostContent({ post }) {
  const router = useRouter();
  const { slug } = router.query;
  const createdAt = typeof post?.createdAt === 'number' ? new Date(post.createdAt) : post.createdAt.toDate();
  const postRef = firestore.collection('users').doc('emojik'.uid).collection('posts').doc(slug);

  function DeletePostButton({ postRef }) {
    const router = useRouter();
  
    const deletePost = async () => {
        await postRef.delete();
        router.push('/');
        toast('post deleted ', { icon: '🗑️' });
    };
  
    return (
      <button className="block-button" onClick={deletePost}>
        Delete
      </button>
    );
  }
  

  return (
    <div className="block-l">
      <h1>{post?.title}</h1>
      <span className="block-text">
        Written by{' '}
        <Link href={`/${post.username}/`}>
          <a className="block-text lc">@{post.username}</a>
        </Link>{' '}
        on {createdAt.toISOString()}
      </span>
      <ReactMarkdown>{post?.content}</ReactMarkdown>
      <DeletePostButton postRef={postRef} />
    </div>
  );
}
