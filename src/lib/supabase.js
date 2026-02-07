import { createClient } from '@supabase/supabase-js';
import { SUPABASE_URL, SUPABASE_ANON_KEY } from '../config';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

/**
 * Sign in a bank user with email + password.
 */
export async function signInBank(email, password) {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });
  if (error) throw error;
  return data;
}

/**
 * Sign out current user.
 */
export async function signOutBank() {
  const { error } = await supabase.auth.signOut();
  if (error) throw error;
}

/**
 * Get current session.
 */
export async function getSession() {
  const { data, error } = await supabase.auth.getSession();
  if (error) throw error;
  return data.session;
}

/**
 * Fetch bank profile by email.
 */
export async function fetchBankProfile(email) {
  const { data, error } = await supabase
    .from('banks')
    .select('*')
    .eq('email', email)
    .single();
  if (error) throw error;
  return data;
}

/**
 * Insert a transaction record after approval.
 */
export async function insertTransaction(transaction) {
  const { data, error } = await supabase
    .from('transactions')
    .insert(transaction)
    .select()
    .single();
  if (error) throw error;
  return data;
}

/**
 * Fetch transaction history for a specific bank.
 */
export async function fetchTransactions(bankId) {
  const { data, error } = await supabase
    .from('transactions')
    .select('*')
    .eq('initiator_bank_id', bankId)
    .order('created_at', { ascending: false });
  if (error) throw error;
  return data || [];
}

/**
 * Log a session action.
 */
export async function logSessionAction(bankId, action, metadata = {}) {
  await supabase.from('session_log').insert({
    bank_id: bankId,
    action,
    metadata,
  });
}
